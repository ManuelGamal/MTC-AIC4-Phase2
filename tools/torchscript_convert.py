"""Export a TorchScript-friendly UETrack encoder wrapper.

This script verifies that the model can be loaded and that the encoder path can be
traced with representative tensor inputs. The decoder path is intentionally left
as a best-effort runtime module because the current implementation uses Python
container logic that is not stable under TorchScript tracing.

Usage:
    python tools/torchscript_convert.py --device cpu [--fp16]

The script saves `tools/uetrack_encoder_wrapper.pt` when tracing succeeds.
"""
import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so imports like `predictor` resolve
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import torch
from predictor import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    p.add_argument('--fp16', action='store_true')
    args = p.parse_args()

    model_bundle = load_model(device=args.device)
    network = model_bundle.network
    cfg = model_bundle.cfg

    network.eval()

    if args.fp16:
        try:
            network.half()
            print('Converted network weights to FP16')
        except Exception as e:
            print('Failed to convert network to FP16:', e)

    # Build representative sample inputs
    num_templates = max(1, getattr(cfg, 'num_template', 2))
    template_size = getattr(cfg, 'template_size', 112)
    search_size = getattr(cfg, 'search_size', 224)

    dtype = torch.float16 if args.fp16 else torch.float32
    device = torch.device(args.device)

    # Create a single template tensor and replicate
    template = torch.randn(1, 3, template_size, template_size, dtype=dtype, device=device)
    template_list = [template for _ in range(2)]

    search = torch.randn(1, 3, search_size, search_size, dtype=dtype, device=device)
    search_list = [search]

    # Create dummy template annotation tensors of expected shape (1, 4) normalized
    template_anno = torch.zeros((1,4), dtype=dtype, device=device)
    template_anno_list = [template_anno for _ in range(2)]

    # Try a forward pass to ensure it runs
    with torch.no_grad():
        try:
            feat = network.forward_encoder(template_list, search_list, template_anno_list, text_src=None, task_index=0)
            print('forward_encoder executed; output type:', type(feat))
        except Exception as e:
            print('forward_encoder failed during test run:', e)
            return

        # If forward_encoder returns a tuple/list, pick first element as features
        try:
            features = feat[0] if isinstance(feat, (list, tuple)) else feat
        except Exception:
            features = feat

        # Try forward_decoder
        try:
            out = network.forward_decoder(features)
            print('forward_decoder executed; keys:', list(out.keys()) if isinstance(out, dict) else type(out))
        except Exception as e:
            print('forward_decoder failed during test run:', e)
            return

    def _summarize_output(obj):
        if isinstance(obj, torch.Tensor):
            return f"Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype})"
        if isinstance(obj, (list, tuple)):
            return f"{type(obj).__name__}(len={len(obj)})"
        if isinstance(obj, dict):
            return f"dict(keys={list(obj.keys())})"
        return type(obj).__name__

    # Tracing encoder wrapper
    try:
        # Build tensor versions of inputs: stack templates and annos to pure tensors
        templates_tensor = torch.cat(template_list, dim=0)  # shape (T, C, H, W)
        # templates_tensor currently is list of (1,C,H,W) -> cat gives (T, C, H, W) if dim=0
        # but ensure shape is (T, C, H, W)
        templates_tensor = templates_tensor.squeeze(1) if templates_tensor.dim() == 5 and templates_tensor.size(1) == 1 else templates_tensor
        search_tensor = search_list[0]
        # stack annos
        template_annos_tensor = torch.cat(template_anno_list, dim=0)

        # Define wrapper modules that accept plain tensors
        import torch.nn as nn

        class EncoderWrapper(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, templates, search, annos):
                # templates: (T, C, H, W); convert to list of (1,C,H,W)
                tl = [templates[i].unsqueeze(0) for i in range(templates.shape[0])]
                sl = [search]
                ta = [annos[i].unsqueeze(0) for i in range(annos.shape[0])]
                return self.net.forward_encoder(tl, sl, ta, None, 0)

        class DecoderWrapper(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, features):
                return self.net.forward_decoder(features)

        enc_wrapper = EncoderWrapper(network)

        print('Tracing encoder wrapper...')
        traced_enc = torch.jit.trace(enc_wrapper, (templates_tensor, search_tensor, template_annos_tensor))
        enc_path = 'tools/uetrack_encoder_wrapper.pt'
        traced_enc.save(enc_path)
        print('Saved traced encoder wrapper to', enc_path)
    except Exception as e:
        print('trace/trace_module failed:', e)

    print('Encoder output summary:', _summarize_output(feat))
    print('Decoder summary:', _summarize_output(out))
    print('Decoder TorchScript export skipped: current UETrack decoder uses container logic that is not trace-safe.')


if __name__ == '__main__':
    main()
