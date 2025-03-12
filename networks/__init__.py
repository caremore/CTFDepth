import torch.nn as nn
from functools import partial

from .resnet_encoder import ResnetEncoder, ResnetEncoderMatching,PoseResnetEncoder
from .depth_decoder import DepthDecoder
from .pose_decoder import PoseDecoder
from .pose_cnn import PoseCNN

# Define a Swin Transformer for SimMIM
from .simmim import build_simmim

simmim_config = {
    'model_type': 'swin',
    'img_size': (192,640),#(480,640)
    'patch_size': 4,
    'in_chans': 3,
    'num_classes': 0,
    'embed_dim': 128,
    'depths': [2, 2, 18, 2],
    'num_heads': [4, 8, 16, 32],
    'window_size': 2,#5
    'mlp_ratio': 4.,
    'qkv_bias': True,
    'qk_scale': None,
    'drop_rate': 0.0,
    'drop_path_rate': 0.1,
    'ape': False,
    'patch_norm': True,
    'use_checkpoint': False,
}
vitmim_config = {
    'img_size': 224,
    'patch_size': 4,
    'in_chans': 3,
    'num_classes': 0,
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
    'mlp_ratio': 4.0,
    'qkv_bias': True,
    'drop_rate': 0.0,
    'drop_path_rate': 0.1,
    'norm_layer': partial(nn.LayerNorm, eps=1e-6),
    'init_values': None,
    'use_abs_pos_emb': False,
    'use_rel_pos_bias': False,
    'use_shared_rel_pos_bias': False,
    'use_mean_pooling': False,
}

SimmimMode = build_simmim(simmim_config,pretrained=True)