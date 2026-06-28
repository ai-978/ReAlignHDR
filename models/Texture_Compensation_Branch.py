import torch
import torch.nn as nn
try:
    from .Source_Guided_Attention import SourceGuidedAttention
    from .Flow_Network import FlowNetwork, warp
    from .Gated_Fusion_Projection import GatedFusionProjection2CToC
except ImportError:  # allow importing this file directly
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from Source_Guided_Attention import SourceGuidedAttention
    from Flow_Network import FlowNetwork, warp
    from Gated_Fusion_Projection import GatedFusionProjection2CToC


class TextureCompensationBranch(nn.Module):
    def __init__(self, feature_dim=None, img_size=(128, 128),
                 depth=None, num_heads=None, window_size=None,
                 drop_path_rate=0., drop_path=None, qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., use_checkpoint=False):
        super(TextureCompensationBranch, self).__init__()

        if feature_dim is None:
            raise ValueError("feature_dim must be specified explicitly.")
        if depth is None:
            raise ValueError("depth must be specified explicitly.")
        if num_heads is None:
            raise ValueError("num_heads must be specified explicitly.")
        if window_size is None:
            raise ValueError("window_size must be specified explicitly.")

        self.dim = feature_dim
        if isinstance(img_size, int):
            self.h = self.w = img_size
        else:
            self.h, self.w = img_size

        self.flow_network = FlowNetwork(in_channels=feature_dim)
        
        self.fuser = GatedFusionProjection2CToC(self.dim)

        if isinstance(depth, int):
            depths = (depth, depth, depth, depth)
        else:
            depths = tuple(depth)

        if isinstance(num_heads, int):
            num_heads_list = (num_heads, num_heads, num_heads, num_heads)
        else:
            num_heads_list = tuple(num_heads)
        if len(num_heads_list) == 1 and len(depths) > 1:
            num_heads_list = num_heads_list * len(depths)
        if len(num_heads_list) != len(depths):
            raise ValueError(f"num_heads must have length {len(depths)}, got {len(num_heads_list)}")

        embed_dim = feature_dim * 3

        self.cross_attn_path1 = SourceGuidedAttention(
            img_size=(self.h, self.w),
            patch_size=1,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads_list,
            window_size=window_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop,
            attn_drop_rate=attn_drop,
            drop_path_rate=drop_path_rate,
            drop_path=drop_path,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=use_checkpoint
        )

        self.cross_attn_path2 = SourceGuidedAttention(
            img_size=(self.h, self.w),
            patch_size=1,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads_list,
            window_size=window_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop,
            attn_drop_rate=attn_drop,
            drop_path_rate=drop_path_rate,
            drop_path=drop_path,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=use_checkpoint
        )

    def forward(self, x1, x2, x3, ldr1, ldr2, ldr3, m1, m2, m3):
        B, C, H, W = x1.shape
        x_size = (H, W)

        # Eq. (7): estimate f_{1->2}, f_{3->2} from shallow features F1, F2, F3.
        flow_1_2, flow_3_2 = self.flow_network(x1, x2, x3)

        x1_warp = warp(x1, flow_1_2)
        x3_warp = warp(x3, flow_3_2)

        x1_warp_flat = x1_warp.flatten(2).transpose(1, 2)
        x3_warp_flat = x3_warp.flatten(2).transpose(1, 2)
        x2_flat      = x2.flatten(2).transpose(1, 2)

        t1_flat = self.cross_attn_path1(x=x2_flat, y=x1_warp_flat, x_size=x_size)
        t3_flat = self.cross_attn_path2(x=x2_flat, y=x3_warp_flat, x_size=x_size)

        t1_out = t1_flat.transpose(1, 2).view(B, C, H, W)
        t3_out = t3_flat.transpose(1, 2).view(B, C, H, W)

        fused_img = self.fuser(t1_out, t3_out)  # (B, C, H, W)

        fused_out = fused_img.flatten(2).transpose(1, 2).contiguous()  # (B, H*W, C)

        return fused_out, flow_1_2, flow_3_2
