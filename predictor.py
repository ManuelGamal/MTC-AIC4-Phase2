"""
Phase-2 UETrack predictor adapter.

The organizer-owned runner imports only ``load_model`` and ``run_tracker`` from
this file.  All preprocessing, model construction, online tracking, and
postprocessing needed by the original UETrack submission live here.
"""

from __future__ import annotations

import math
import os
import re
import sys
import types
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent
UETRACK_ROOT = REPO_ROOT / "UETrack"
CHECKPOINT_CANDIDATES = (
    REPO_ROOT / "checkpoints" / "model_final.pth",
    REPO_ROOT / "checkpoints" / "model.pth",
)

IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
USE_CHANNELS_LAST = os.getenv("ORIN_CHANNELS_LAST", "1") == "1"
USE_CUDA_AUTOTUNE = os.getenv("ORIN_CUDA_AUTOTUNE", "1") == "1"
USE_TF32 = os.getenv("ORIN_TF32", "0") == "1"
USE_CUDA_WARMUP = os.getenv("ORIN_CUDA_WARMUP", "1") == "1"


@dataclass
class RuntimeConfig:
    seed: int = 42
    uetrack_variant: str = "uetrack_base"
    template_size: int = 112
    search_size: int = 224
    template_factor: float = 2.0
    search_factor: float = 4.5
    update_interval: int = 25
    num_templates: int = 2
    enter_absent_conf: float = 0.25
    exit_absent_conf: float = 0.40
    enter_streak: int = 2
    exit_streak: int = 3


@dataclass
class UETrackModelBundle:
    network: torch.nn.Module
    cfg: RuntimeConfig
    device: torch.device


def load_model(device: str = "cuda") -> UETrackModelBundle:
    """
    Build UETrack and load the original checkpoint once before evaluation.
    """

    _ensure_runtime_ready()
    runtime_cfg = RuntimeConfig()
    torch_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")

    _seed_everything(runtime_cfg.seed)
    _configure_inference_backends(torch_device)

    checkpoint_path = _find_checkpoint()

    from lib.config.uetrack.config import cfg as uetrack_cfg
    from lib.config.uetrack.config import update_config_from_file
    from lib.models.uetrack import build_uetrack_inference

    exp_file = UETRACK_ROOT / "experiments" / "uetrack" / f"{runtime_cfg.uetrack_variant}.yaml"
    update_config_from_file(str(exp_file))

    uetrack_cfg.MODEL.ENCODER.PRETRAIN_TYPE = None
    uetrack_cfg.DATA.MULTI_MODAL_LANGUAGE = False
    uetrack_cfg.DATA.MULTI_MODAL_VISION = False
    uetrack_cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT = False
    uetrack_cfg.TEST.MULTI_MODAL_VISION.DEFAULT = False
    uetrack_cfg.TEST.NUM_TEMPLATES = runtime_cfg.num_templates

    # Temporarily monkey-patch torch.load to prevent crashes when loading None
    _orig_load = torch.load
    def _safe_load(f, *args, **kwargs):
        if f is None or (isinstance(f, (str, Path)) and str(f) == ''):
            return {}
        if isinstance(f, (str, Path)) and not Path(str(f)).exists() and str(f) != str(checkpoint_path):
            return {}
        return _orig_load(f, *args, **kwargs)

    torch.load = _safe_load
    try:
        network = build_uetrack_inference(uetrack_cfg)
    finally:
        torch.load = _orig_load
    _replace_six_channel_patch_embed(network)
    _load_tracking_checkpoint(network, checkpoint_path, uetrack_cfg)

    if os.getenv("ORIN_QUANTIZE_DYNAMIC", "0") == "1" and torch_device.type == "cpu":
        network = _apply_dynamic_quantization(network)

    network = network.to(torch_device).eval()
    if torch_device.type == "cuda" and USE_CHANNELS_LAST:
        network = network.to(memory_format=torch.channels_last)
    if torch_device.type == "cuda" and USE_CUDA_WARMUP:
        _warmup_network(network, torch_device, runtime_cfg)

    return UETrackModelBundle(network=network, cfg=runtime_cfg, device=torch_device)


def run_tracker(
    model: UETrackModelBundle,
    video_path: str | Path,
    init_box_path: str | Path,
) -> List[dict]:
    """
    Run UETrack on one sequence and return one prediction per frame.
    """

    init_box = read_init_box(init_box_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path!s}")

    predictions: List[dict] = []
    tracker = UETrackOnline(model.network, model.cfg, model.device)
    frame_idx = 0

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if frame_idx == 0:
                tracker.initialize(frame_rgb, init_box)
                pred_box = init_box
            else:
                pred_box = tracker.track(frame_rgb)

            height, width = frame_rgb.shape[:2]
            pred_box = clip_bbox(pred_box, width, height)
            predictions.append(format_prediction(frame_idx, pred_box))
            frame_idx += 1
    finally:
        capture.release()
        if model.device.type == "cuda":
            torch.cuda.empty_cache()

    return predictions


class UETrackOnline:
    """
    Online tracker wrapper matching the original competition pipeline.
    """

    def __init__(self, network: torch.nn.Module, cfg: RuntimeConfig, device: torch.device):
        from lib.test.utils.hann import hann2d

        self.network = network
        self.cfg = cfg
        self.device = device
        self.template_size = cfg.template_size
        self.search_size = cfg.search_size
        self.template_factor = cfg.template_factor
        self.search_factor = cfg.search_factor
        self.update_intervals = cfg.update_interval
        self.num_template = cfg.num_templates
        self.memory_bank = 50
        self.update_threshold = 0.50
        self.window_influence = 0.40
        self.box_ema_alpha = 0.25
        self.enter_absent_conf = cfg.enter_absent_conf
        self.exit_absent_conf = cfg.exit_absent_conf
        self.enter_streak = cfg.enter_streak
        self.exit_streak = cfg.exit_streak
        self.min_area_ratio = 8e-5
        self.area_low_conf = 0.20
        self.border_low_conf = 0.10
        self.state = None
        self.frame_id = 0
        self.absent = False
        self.lost_streak = 0
        self.recover_streak = 0
        self.vx = 0.0
        self.vy = 0.0
        self.last_cx = 0.0
        self.last_cy = 0.0
        self._ema_w = None
        self._ema_h = None

        feat_size = self.search_size // 16
        self.output_window = hann2d(torch.tensor([feat_size, feat_size]).long(), centered=True).to(device)

    def initialize(self, image: np.ndarray, init_box: Sequence[float]) -> None:
        height, width = image.shape[:2]
        self.state = clip_bbox(init_box, width, height)
        z_patch_arr, resize_factor = sample_target(
            image,
            self.state,
            self.template_factor,
            output_sz=self.template_size,
        )
        template = image_to_device_tensor(z_patch_arr, self.device)
        prev_box_crop = transform_image_to_crop(
            torch.tensor(self.state),
            torch.tensor(self.state),
            resize_factor,
            torch.tensor([self.template_size, self.template_size]),
            normalize=True,
        )
        template_anno = prev_box_crop.to(self.device).unsqueeze(0)

        self.template_list = [template] * self.num_template
        self.template_anno_list = [template_anno] * self.num_template
        self.memory_template_list = self.template_list.copy()
        self.memory_template_anno_list = self.template_anno_list.copy()

        self.frame_id = 0
        self.absent = False
        self.lost_streak = 0
        self.recover_streak = 0
        self.vx = 0.0
        self.vy = 0.0
        self.last_cx = self.state[0] + 0.5 * self.state[2]
        self.last_cy = self.state[1] + 0.5 * self.state[3]
        self._ema_w = self.state[2]
        self._ema_h = self.state[3]

    def track(self, image: np.ndarray) -> List[float]:
        height, width = image.shape[:2]
        self.frame_id += 1

        if self.absent or self.lost_streak > 0:
            self.state[0] += self.vx
            self.state[1] += self.vy
            self.state = clip_box(self.state, height, width, margin=10)

        current_search_factor = self.search_factor
        if self.lost_streak > 0 or self.absent:
            current_search_factor = min(self.search_factor * 1.5, 6.0)

        x_patch_arr, resize_factor = sample_target(
            image,
            self.state,
            current_search_factor,
            output_sz=self.search_size,
        )
        search = image_to_device_tensor(x_patch_arr, self.device)

        autocast_ctx = (
            torch.autocast(device_type="cuda")
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            raw = self.network.forward_encoder(
                self.template_list,
                [search],
                self.template_anno_list,
                text_src=None,
                task_index=0,
            )
            features = raw[0]
            out_dict = self.network.forward_decoder(features)

        response = (
            (1 - self.window_influence) * out_dict["score_map"]
            + self.window_influence * self.output_window
        )
        if "size_map" in out_dict:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(
                response,
                out_dict["size_map"],
                out_dict["offset_map"],
                return_score=True,
            )
        else:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(
                response,
                out_dict["offset_map"],
                return_score=True,
            )

        pred_boxes = pred_boxes.view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.search_size / resize_factor).tolist()
        conf = float(conf_score.item()) if hasattr(conf_score, "item") else float(conf_score)

        self.state = clip_box(self.map_box_back(pred_box, resize_factor), height, width, margin=10)
        self._smooth_box()
        self._update_absence_state(conf, width, height)

        if self.absent:
            return [0.0, 0.0, 0.0, 0.0]

        self._update_templates(image, conf)
        self._update_motion(conf)
        return self.state

    def map_box_back(self, pred_box: Sequence[float], resize_factor: float) -> List[float]:
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def _smooth_box(self) -> None:
        if self._ema_w is None or self._ema_h is None:
            self._ema_w = self.state[2]
            self._ema_h = self.state[3]
            return
        self._ema_w = self.box_ema_alpha * self._ema_w + (1 - self.box_ema_alpha) * self.state[2]
        self._ema_h = self.box_ema_alpha * self._ema_h + (1 - self.box_ema_alpha) * self.state[3]
        self.state = [self.state[0], self.state[1], self._ema_w, self._ema_h]

    def _update_absence_state(self, conf: float, width: int, height: int) -> None:
        area_ratio = (self.state[2] * self.state[3]) / (width * height + 1e-6)
        near_border = (
            self.state[0] <= 2
            or self.state[1] <= 2
            or self.state[0] + self.state[2] >= width - 2
            or self.state[1] + self.state[3] >= height - 2
        )
        bad = (
            conf < self.enter_absent_conf
            or (area_ratio < self.min_area_ratio and conf < self.area_low_conf)
            or (near_border and conf < self.border_low_conf)
        )
        if bad:
            self.lost_streak += 1
            self.recover_streak = 0
        else:
            self.lost_streak = 0
            if self.absent and conf >= self.exit_absent_conf and area_ratio >= self.min_area_ratio:
                self.recover_streak += 1
            else:
                self.recover_streak = 0

        if not self.absent and self.lost_streak >= self.enter_streak:
            self.absent = True
        if self.absent and self.recover_streak >= self.exit_streak:
            self.absent = False
            self.lost_streak = 0
            self.recover_streak = 0

    def _update_templates(self, image: np.ndarray, conf: float) -> None:
        if self.num_template <= 1:
            return

        if conf > self.update_threshold:
            z_patch_arr, resize_factor = sample_target(
                image,
                self.state,
                self.template_factor,
                output_sz=self.template_size,
            )
            dyn_template = preprocess_image(z_patch_arr).unsqueeze(0)
            if self.device.type == "cuda" and USE_CHANNELS_LAST:
                dyn_template = dyn_template.contiguous(memory_format=torch.channels_last)
            prev_box_crop = transform_image_to_crop(
                torch.tensor(self.state),
                torch.tensor(self.state),
                resize_factor,
                torch.tensor([self.template_size, self.template_size]),
                normalize=True,
            )
            dyn_anno = prev_box_crop.to(self.device).unsqueeze(0)

            self.memory_template_list.append(dyn_template)
            self.memory_template_anno_list.append(dyn_anno)
            if len(self.memory_template_list) > self.memory_bank:
                self.memory_template_list.pop(0)
                self.memory_template_anno_list.pop(0)

        if self.frame_id % self.update_intervals == 0 and not self.absent:
            len_list = len(self.memory_template_anno_list)
            interval = len_list // self.num_template
            for i in range(1, self.num_template):
                idx = min(interval * i, len_list - 1)
                self.template_list[i] = self.memory_template_list[idx].to(self.device)
                self.template_anno_list[i] = self.memory_template_anno_list[idx].to(self.device)

    def _update_motion(self, conf: float) -> None:
        curr_cx = self.state[0] + 0.5 * self.state[2]
        curr_cy = self.state[1] + 0.5 * self.state[3]
        if conf > self.update_threshold and not self.absent:
            self.vx = 0.8 * self.vx + 0.2 * (curr_cx - self.last_cx)
            self.vy = 0.8 * self.vy + 0.2 * (curr_cy - self.last_cy)
        self.last_cx = curr_cx
        self.last_cy = curr_cy


def _ensure_runtime_ready() -> None:
    if not UETRACK_ROOT.exists():
        raise FileNotFoundError(
            "UETrack source is missing. Vendor UETrack/ at the repository root before inference."
        )
    if str(UETRACK_ROOT) not in sys.path:
        sys.path.insert(0, str(UETRACK_ROOT))
    _patch_uetrack_files()
    _install_clip_stub()


def _patch_uetrack_files() -> None:
    # 0. Patch fastitpn_teacher.py: 6-channel -> 3-channel
    fp_teacher = UETRACK_ROOT / 'lib/models/uetrack/fastitpn_teacher.py'
    if fp_teacher.exists():
        content = fp_teacher.read_text(encoding='utf-8')
        if 'in_chans=6' in content:
            fp_teacher.write_text(content.replace('in_chans=6', 'in_chans=3'), encoding='utf-8')

    # 1. Patch fastitpn.py to prevent the KeyError on empty checkpoints
    fp = UETRACK_ROOT / 'lib/models/uetrack/fastitpn.py'
    if fp.exists():
        content = fp.read_text(encoding='utf-8')
        old = (
            "    else:\n"
            "        state_dict = checkpoint\n"
            "    pe = state_dict['pos_embed'].float()"
        )
        new = (
            "    else:\n"
            "        state_dict = checkpoint\n"
            "    if not state_dict:  # empty checkpoint guard\n"
            "        return\n"
            "    pe = state_dict['pos_embed'].float()"
        )
        if old in content:
            fp.write_text(content.replace(old, new), encoding='utf-8')
    
    # 2. Patch uetrack.py to prevent text_encoder crashes
    fp_ue = UETRACK_ROOT / 'lib/models/uetrack/uetrack.py'
    if fp_ue.exists():
        content_ue = fp_ue.read_text(encoding='utf-8')
        old_ue = (
            "self.interface_text_proj = nn.Linear(self.text_encoder.textencoder_dim, self.encoder.num_channels)\n"
            "        trunc_normal_(self.interface_text_proj.weight, std=.02)\n"
            "        nn.init.constant_(self.interface_text_proj.bias, 0)"
        )
        new_ue = (
            "if self.text_encoder is not None:\n"
            "            self.interface_text_proj = nn.Linear(self.text_encoder.textencoder_dim, self.encoder.num_channels)\n"
            "            trunc_normal_(self.interface_text_proj.weight, std=.02)\n"
            "            nn.init.constant_(self.interface_text_proj.bias, 0)\n"
            "        else:\n"
            "            self.interface_text_proj = None"
        )
        if old_ue in content_ue:
            fp_ue.write_text(content_ue.replace(old_ue, new_ue), encoding='utf-8')

def _install_clip_stub() -> None:
    if "clip" in sys.modules:
        return

    clip_stub = types.ModuleType("clip")

    def _missing_clip_load(*_args, **_kwargs):
        raise RuntimeError("CLIP text encoder is disabled for this submission.")

    def _tokenize(text):
        del text
        return torch.zeros((1, 77), dtype=torch.long)

    clip_stub.load = _missing_clip_load
    clip_stub.tokenize = _tokenize
    sys.modules["clip"] = clip_stub


def _apply_dynamic_quantization(network: torch.nn.Module) -> torch.nn.Module:
    """
    Apply post-training dynamic quantization to linear layers only.

    This keeps the default submission path unchanged and can be enabled via
    ORIN_QUANTIZE_DYNAMIC=1 for CPU-side quantization experiments.
    """

    quantizable_types = {nn.Linear}
    try:
        return torch.quantization.quantize_dynamic(network.cpu(), quantizable_types, dtype=torch.qint8)
    except Exception:
        return network




def _find_checkpoint() -> Path:
    for checkpoint_path in CHECKPOINT_CANDIDATES:
        if checkpoint_path.exists():
            return checkpoint_path
    candidates = ", ".join(str(path.relative_to(REPO_ROOT)) for path in CHECKPOINT_CANDIDATES)
    raise FileNotFoundError(
        f"UETrack checkpoint not found. Expected one of: {candidates}. "
        "Run python download.py or place model_final.pth under checkpoints/."
    )


def _replace_six_channel_patch_embed(network: torch.nn.Module) -> None:
    for _module_name, module in list(network.named_modules()):
        if (
            "patch_embed" in _module_name
            and hasattr(module, "proj")
            and isinstance(module.proj, nn.Conv2d)
            and module.proj.in_channels == 6
        ):
            old = module.proj
            module.proj = nn.Conv2d(
                3,
                old.out_channels,
                old.kernel_size,
                old.stride,
                old.padding,
                bias=(old.bias is not None),
            )


def _load_tracking_checkpoint(
    network: torch.nn.Module,
    checkpoint_path: Path,
    uetrack_cfg,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("net", checkpoint.get("model", checkpoint.get("state_dict", checkpoint)))
    state_dict = dict(state_dict)

    for key in list(state_dict.keys()):
        value = state_dict[key]
        if "patch_embed" in key and "proj.weight" in key and getattr(value, "ndim", 0) == 4 and value.shape[1] == 6:
            state_dict[key] = value[:, :3, :, :] + value[:, 3:, :, :]

    _fix_positional_embeddings(state_dict, network, uetrack_cfg)
    network.load_state_dict(state_dict, strict=False)


def _fix_positional_embeddings(state_dict: dict, network: torch.nn.Module, uetrack_cfg) -> None:
    stride = uetrack_cfg.MODEL.ENCODER.STRIDE
    num_template_tokens = (uetrack_cfg.TEST.TEMPLATE_SIZE // stride) ** 2
    num_search_tokens = (uetrack_cfg.TEST.SEARCH_SIZE // stride) ** 2
    model_state = network.state_dict()

    for key in list(state_dict.keys()):
        if "pos_embed" not in key or key not in model_state:
            continue
        ckpt_pe = state_dict[key]
        model_pe = model_state[key]
        if ckpt_pe.shape == model_pe.shape:
            continue

        ckpt_tokens = ckpt_pe.shape[1]
        model_tokens = model_pe.shape[1]
        dim = ckpt_pe.shape[2]
        if ckpt_tokens == num_template_tokens + num_search_tokens and model_tokens == 2 * num_template_tokens + num_search_tokens:
            template_pe = ckpt_pe[:, :num_template_tokens, :]
            search_pe = ckpt_pe[:, num_template_tokens:, :]
            state_dict[key] = torch.cat([template_pe, template_pe, search_pe], dim=1)
        elif int(round(math.sqrt(ckpt_tokens))) ** 2 == ckpt_tokens and int(round(math.sqrt(model_tokens))) ** 2 == model_tokens:
            ckpt_side = int(round(math.sqrt(ckpt_tokens)))
            model_side = int(round(math.sqrt(model_tokens)))
            pe = ckpt_pe.reshape(1, ckpt_side, ckpt_side, dim).permute(0, 3, 1, 2).float()
            pe = F.interpolate(pe, size=(model_side, model_side), mode="bicubic", align_corners=False)
            state_dict[key] = pe.permute(0, 2, 3, 1).reshape(1, model_tokens, dim)
        else:
            del state_dict[key]


def read_init_box(path: str | Path) -> List[float]:
    with open(path, "r", encoding="utf-8") as handle:
        line = handle.readline().strip()
    parts = [float(part) for part in re.split(r"[\s,]+", line) if part]
    if len(parts) < 4:
        raise ValueError(f"Expected at least four bbox values in {path!s}")
    return parts[:4]


def sample_target(image: np.ndarray, bbox: Sequence[float], factor: float, output_sz: int):
    if hasattr(bbox, "tolist"):
        x, y, w, h = bbox.tolist()
    else:
        x, y, w, h = bbox

    crop_size = math.ceil(math.sqrt(w * h) * factor)
    if crop_size < 1:
        raise ValueError("Too small bounding box.")

    x1 = round(x + 0.5 * w - crop_size * 0.5)
    x2 = x1 + crop_size
    y1 = round(y + 0.5 * h - crop_size * 0.5)
    y2 = y1 + crop_size

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - image.shape[1] + 1, 0)
    y1_pad = max(0, -y1)
    y2_pad = max(y2 - image.shape[0] + 1, 0)

    crop = image[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    crop = cv2.copyMakeBorder(crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)

    resize_factor = output_sz / crop_size
    crop = cv2.resize(crop, (output_sz, output_sz))
    return crop, resize_factor


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    image = image.astype(np.float32) / 255.0
    image = (image - IMG_MEAN) / IMG_STD
    image = image.transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(image)).float()


def image_to_device_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = preprocess_image(image).unsqueeze(0)
    if device.type == "cuda" and USE_CHANNELS_LAST:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor.to(device, non_blocking=True)


def transform_image_to_crop(
    box_in: torch.Tensor,
    box_extract: torch.Tensor,
    resize_factor: float,
    crop_sz: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]
    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]
    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor
    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    if normalize:
        return box_out / (crop_sz[0] - 1)
    return box_out


def clip_box(box: Sequence[float], height: int, width: int, margin: float = 0) -> List[float]:
    x1, y1, w, h = [float(value) for value in box[:4]]
    x2 = x1 + w
    y2 = y1 + h
    x1 = min(max(0.0, x1), width - margin)
    x2 = min(max(margin, x2), width)
    y1 = min(max(0.0, y1), height - margin)
    y2 = min(max(margin, y2), height)
    return [x1, y1, max(margin, x2 - x1), max(margin, y2 - y1)]


def clip_bbox(bbox: Sequence[float], width: int, height: int) -> List[float]:
    x, y, w, h = [float(value) for value in bbox[:4]]
    x = max(0.0, min(x, max(0.0, width - 1.0)))
    y = max(0.0, min(y, max(0.0, height - 1.0)))
    w = max(0.0, min(w, width - x))
    h = max(0.0, min(h, height - y))
    return [x, y, w, h]


def format_prediction(frame_idx: int, bbox: Sequence[float]) -> dict:
    x, y, w, h = [float(value) for value in bbox[:4]]
    return {
        "frame_idx": int(frame_idx),
        "x": round(x, 3),
        "y": round(y, 3),
        "w": round(w, 3),
        "h": round(h, 3),
    }


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = USE_CUDA_AUTOTUNE


def _configure_inference_backends(device: torch.device) -> None:
    torch.set_grad_enabled(False)
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = USE_CUDA_AUTOTUNE
    torch.backends.cudnn.allow_tf32 = USE_TF32
    torch.backends.cuda.matmul.allow_tf32 = USE_TF32


def _warmup_network(network: torch.nn.Module, device: torch.device, cfg: RuntimeConfig) -> None:
    template = torch.zeros((1, 3, cfg.template_size, cfg.template_size), device=device)
    search = torch.zeros((1, 3, cfg.search_size, cfg.search_size), device=device)
    if USE_CHANNELS_LAST:
        template = template.contiguous(memory_format=torch.channels_last)
        search = search.contiguous(memory_format=torch.channels_last)
    template_anno = torch.tensor([[0.25, 0.25, 0.5, 0.5]], device=device)
    autocast_ctx = torch.autocast(device_type="cuda")
    try:
        with torch.inference_mode(), autocast_ctx:
            raw = network.forward_encoder(
                [template, template],
                [search],
                [template_anno, template_anno],
                text_src=None,
                task_index=0,
            )
            network.forward_decoder(raw[0])
        torch.cuda.synchronize(device)
    except Exception:
        pass
