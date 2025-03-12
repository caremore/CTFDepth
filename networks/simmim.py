# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------



import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .swin_transformer import SwinTransformer
from .vision_transformer import VisionTransformer


class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)

    def forward(self, x, mask):
        x = self.patch_embed(x)
        # torch.Size([3, 19200, 128])
        assert mask is not None
        B, L, _ = x.shape

        mask_tokens = self.mask_token.expand(B, L, -1)

        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1. - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        #torch.Size([3, 1024, 300])
        #H = W = int(L ** 0.5)
        H = 6#15
        W = 20#20
        x = x.reshape(B, C, H, W)
        return x,features

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class VisionTransformerForSimMIM(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)


        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x_aug,x_clear, mask):

        outputs = {}
        x_aug = (x_aug - 0.45) / 0.225
        x_clear = (x_clear - 0.45) / 0.225

        B1, C1, H1, W1 = x_aug.shape
        if mask is None:
            B, H, W = B1, H1 // 4, W1 // 4
            mask = torch.zeros(B, H, W).to(x_clear.device)  # 所有像素不遮挡

        z,features = self.encoder(x_aug, mask)
        x_clear_rec = self.decoder(z)
        #print(x_clear_rec.min(), x_clear_rec.max())



        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x_clear, x_clear_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        outputs[("x_aug")] = x_aug
        outputs[("x_clear")] = x_clear
        outputs[("x_clear_rec")] = x_clear_rec
        outputs[("mask")] = mask

        return loss,outputs,features

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}

def build_simmim(config,pretrained=True):
    model_type = config['model_type']
    if model_type == 'swin':
        encoder = SwinTransformerForSimMIM(
            img_size=config['img_size'],
            patch_size=config['patch_size'],
            in_chans=config['in_chans'],
            num_classes=config['num_classes'],
            embed_dim=config['embed_dim'],
            depths=config['depths'],
            num_heads=config['num_heads'],
            window_size=config['window_size'],
            mlp_ratio=config['mlp_ratio'],
            qkv_bias=config['qkv_bias'],
            qk_scale=config['qk_scale'],
            drop_rate=config['drop_rate'],
            drop_path_rate=config['drop_path_rate'],
            ape=config['ape'],
            patch_norm=config['patch_norm'],
            use_checkpoint=config['use_checkpoint'])
        encoder_stride = 32
    elif model_type == 'vit':
        encoder = VisionTransformerForSimMIM(
            img_size=config['img_size'],
            patch_size=config['patch_size'],
            in_chans=config['in_chans'],
            num_classes=config['num_classes'],
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['num_heads'],
            mlp_ratio=config['mlp_ratio'],
            qkv_bias=config['qkv_bias'],
            drop_rate=config['drop_rate'],
            drop_path_rate=config['drop_path_rate'],
            norm_layer=config['norm_layer'],
            init_values=config['init_values'],
            use_abs_pos_emb=config['use_abs_pos_emb'],
            use_rel_pos_bias=config['use_rel_pos_bias'],
            use_shared_rel_pos_bias=config['use_shared_rel_pos_bias'],
            use_mean_pooling=config['use_mean_pooling'])
        encoder_stride = 16
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride)

    # 导入 ImageNet 的预训练权重
    if pretrained:
        weights = '/root/autodl-fs/pythoncode/allcheng3/newcheng3/simmim_pretrain__swin_base__img192_window6__100ep.pth'
        pretrained_dict = torch.load(weights, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        #print(model)
        # pretrained_model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)
        # model.load_state_dict(pretrained_model.state_dict(), strict=False)

    return model