"""
SUTrack Model
"""
import torch
import math
from torch import nn
import torch.nn.functional as F
from .encoder import build_encoder,build_encoder_teacher
from .clip import build_textencoder
from .decoder import build_decoder
from .task_decoder import build_task_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from timm.models.layers import trunc_normal_

class SUTRACK(nn.Module):
    """ This is the base class for SUTrack """
    def __init__(self, text_encoder, encoder, decoder, task_decoder,
                 num_frames=1, num_template=1,
                 decoder_type="CENTER", task_feature_type="average"):
        """ Initializes the model.
        """
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder_type = decoder_type

        self.class_token = False if (encoder.body.cls_token is None) else True
        self.task_feature_type = task_feature_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.task_decoder = task_decoder
        self.decoder = decoder

        self.num_frames = num_frames
        self.num_template = num_template


    def forward(self, text_data=None,
                template_list=None, search_list=None, template_anno_list=None,
                text_src=None, task_index=None,
                feature=None,mode="encoder"):
        if mode == "text":
            return self.forward_textencoder(text_data)
        elif mode == "encoder":
            return self.forward_encoder(template_list, search_list, template_anno_list, text_src, task_index)
        elif mode == "decoder":
            return self.forward_decoder(feature), self.forward_task_decoder(feature)
        else:
            raise ValueError

    def forward_textencoder(self, text_data):
        # Forward the encoder
        text_src_teacher, text_src = self.text_encoder(text_data)
        return text_src_teacher, text_src

    def forward_encoder(self, template_list, search_list, template_anno_list, text_src, task_index):
        # Forward the encoder
        xz,feature_list = self.encoder(template_list, search_list, template_anno_list, text_src, task_index)
        return xz,feature_list

    def forward_decoder(self, feature, gt_score_map=None):

        feature = feature[0]
        if self.class_token:
            feature = feature[:,1:self.num_patch_x * self.num_frames+1]
        else:
            feature = feature[:,0:self.num_patch_x * self.num_frames] # (B, HW, C)

        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.decoder_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.decoder_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

    def forward_task_decoder(self, feature):
        feature = feature[0]
        if self.task_feature_type == 'class':
            feature = feature[:, 0:1]
        elif self.task_feature_type == 'text':
            feature = feature[:, -1:]
        elif self.task_feature_type == 'average':
            feature = feature.mean(1).unsqueeze(1)
        else:
            raise NotImplementedError('task_feature_type must be choosen from class, text, and average')
        feature = self.task_decoder(feature)
        return feature

class UETrack(nn.Module):
    """ This is the base class for UETrack """
    def __init__(self, text_encoder, encoder, decoder, task_decoder,
                 num_frames=1, num_template=1,
                 decoder_type="CENTER", task_feature_type="average",adjust_layers=None):
        """ Initializes the model.
        """
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder_type = decoder_type

        self.class_token = False if (encoder.body.cls_token is None) else True
        self.task_feature_type = task_feature_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.task_decoder = task_decoder
        self.decoder = decoder

        self.num_frames = num_frames
        self.num_template = num_template

        if self.text_encoder is not None:
            self.interface_text_proj = nn.Linear(self.text_encoder.textencoder_dim, self.encoder.num_channels)
            trunc_normal_(self.interface_text_proj.weight, std=.02)
            nn.init.constant_(self.interface_text_proj.bias, 0)
        else:
            self.interface_text_proj = None
        self.adjust_layers = adjust_layers


    def forward(self, text_data=None,
                template_list=None, search_list=None, template_anno_list=None,
                text_src=None, task_index=None,
                feature=None, student_feature=None,mode="encoder"):
        if mode == "text":
            text_src = text_src.type(self.interface_text_proj.weight.dtype)
            text_src = self.interface_text_proj(text_src).unsqueeze(1)
            return text_src
        elif mode == "encoder":
            return self.forward_encoder(template_list, search_list, template_anno_list, text_src, task_index)
        elif mode == "decoder":
            return self.forward_decoder(feature), self.forward_task_decoder(feature)
        elif mode == "text_dis":
            return self.forward_textencoder_inference(text_data)
        elif mode == "tracking_decoder":
            return self.forward_decoder(feature)
        elif mode == "task_decoder":
            return self.forward_task_decoder(feature)
        else:
            raise ValueError

    def forward_textencoder_inference(self, text_data):
        # Forward the encoder
        text_src_teacher, text_src = self.text_encoder(text_data)
        text_src = text_src.type(self.interface_text_proj.weight.dtype)
        text_src = self.interface_text_proj(text_src).unsqueeze(1)
        return text_src

    def forward_encoder(self, template_list, search_list, template_anno_list, text_src, task_index):
        # Forward the encoder
        xz,feature_list = self.encoder(template_list, search_list, template_anno_list, text_src, task_index)
        if self.adjust_layers is not None:
            feature_list_aligned = []
            for adjust_layer, feat in zip(self.adjust_layers, feature_list):
                feature_list_aligned.append(adjust_layer(feat))
        else:
            feature_list_aligned = feature_list
        return xz,feature_list,feature_list_aligned

    def forward_decoder(self, feature, gt_score_map=None):

        feature = feature[0]
        if self.class_token:
            feature = feature[:,1:self.num_patch_x * self.num_frames+1]
        else:
            feature = feature[:,0:self.num_patch_x * self.num_frames] # (B, HW, C)

        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.decoder_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.decoder_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

    def forward_task_decoder(self, feature):
        feature = feature[0]
        if self.task_feature_type == 'class':
            feature = feature[:, 0:1]
        elif self.task_feature_type == 'text':
            feature = feature[:, -1:]
        elif self.task_feature_type == 'average':
            feature = feature.mean(1).unsqueeze(1)
        else:
            raise NotImplementedError('task_feature_type must be choosen from class, text, and average')
        feature = self.task_decoder(feature)
        return feature


class ADAPTIVE_NET(nn.Module):
    def __init__(self, teacher_enc_chan,student_enc_chan,num_layer,cfg):
        """ Initializes the model.
        """
        super().__init__()
        self.num_layer = num_layer
        self.num_search = int((cfg.DATA.SEARCH.SIZE // cfg.MODEL.ENCODER.STRIDE)**2)
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(teacher_enc_chan+student_enc_chan, student_enc_chan),
                nn.ReLU(),
                nn.Linear(student_enc_chan, 2)   # 输出是否选中
            ) for _ in range(num_layer)
        ])

    def forward(self,feat_t_list,feat_s_list):
        B, _, CS = feat_s_list[0].shape
        B, _, CT = feat_t_list[0].shape
        logits_list = []
        for i in range(self.num_layer):#B,C,H，W
            ft = feat_t_list[i][:,1:1+self.num_search].permute(0,2,1).view(B,CT,int(self.num_search**0.5),int(self.num_search**0.5))
            fs = feat_s_list[i][:,1:1+self.num_search].permute(0,2,1).view(B,CS,int(self.num_search**0.5),int(self.num_search**0.5))
            if ft.dim() == 4:
                ft = F.adaptive_avg_pool2d(ft, 1).flatten(1)
            if fs.dim() == 4:
                fs = F.adaptive_avg_pool2d(fs, 1).flatten(1)
            feat_cat = torch.cat([ft, fs], dim=1)
            logits = self.mlps[i](feat_cat)
            logits_list.append(logits)
            return logits_list


def build_uetrack(cfg):
    #teacher_model
    teacher_encoder = build_encoder_teacher(cfg)
    teacher_decoder = build_decoder(cfg,teacher_encoder)
    teacher_task_decoder = build_task_decoder(cfg,teacher_encoder)
    #student_model
    student_encoder = build_encoder(cfg)
    student_decoder = build_decoder(cfg, student_encoder)
    student_task_decoder = build_task_decoder(cfg, student_encoder)

    #both
    if cfg.DATA.MULTI_MODAL_LANGUAGE:
        text_encoder = build_textencoder(cfg, teacher_encoder)
    else:
        text_encoder = None

    teacher_model = SUTRACK(
        text_encoder,
        teacher_encoder,
        teacher_decoder,
        teacher_task_decoder,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE
    )
    #load teacher model
    teacher_checkpoints = torch.load(cfg.TRAIN.TEACHER_PATH,map_location="cpu")
    state_dict = teacher_checkpoints['net']
    teacher_model.load_state_dict(state_dict,strict=True)
    for p in teacher_model.parameters():
        p.requires_grad = False

    #adjust layer
    if teacher_encoder.num_channels != student_encoder.num_channels:
        adjust_layer = []
        num_adjust_layer = len(cfg.TRAIN.DISTILL_LAYER_T)
        for i in range(num_adjust_layer):
            adjust_layer_i = nn.Linear(student_encoder.num_channels,teacher_encoder.num_channels,bias=False)
            adjust_layer.append(adjust_layer_i)
        adjust_layers =  nn.ModuleList(adjust_layer)
    else:
        adjust_layers = None
    student_model = UETrack(
        text_encoder,
        student_encoder,
        student_decoder,
        student_task_decoder,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE,
        adjust_layers = adjust_layers
    )

    adaptive_layer = ADAPTIVE_NET(
        teacher_enc_chan=teacher_encoder.num_channels,
        student_enc_chan=student_encoder.num_channels,
        num_layer=len(cfg.TRAIN.DISTILL_LAYER_T),
        cfg=cfg
    )
    return teacher_model,student_model,adaptive_layer

def build_uetrack_inference(cfg):
    encoder = build_encoder(cfg)
    if cfg.DATA.MULTI_MODAL_LANGUAGE:
        teacher_encoder = build_encoder_teacher(cfg)
        text_encoder = build_textencoder(cfg, teacher_encoder)
    else:
        text_encoder = None
    decoder = build_decoder(cfg, encoder)
    task_decoder = build_task_decoder(cfg, encoder)
    model = UETrack(
        text_encoder,
        encoder,
        decoder,
        task_decoder,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE
    )

    return model
