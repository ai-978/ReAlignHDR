import torch
import torch.nn as nn
try:
    from .Mask_Guided_Cross_Attention import MaskGuidedCrossAttention
    from .Gated_Fusion_Projection import GatedFusionProjection2CToC
except ImportError:  # allow importing this file directly
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from Mask_Guided_Cross_Attention import MaskGuidedCrossAttention
    from Gated_Fusion_Projection import GatedFusionProjection2CToC

class StructureAnchoringBranch(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 embed_dim=None,
                 depths=None,
                 num_heads=None,
                 window_size=None,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 drop_path=None,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False):
        super().__init__()

        if depths is None:
            raise ValueError("depths must be specified explicitly.")
        if num_heads is None:
            raise ValueError("num_heads must be specified explicitly.")
        if window_size is None:
            raise ValueError("window_size must be specified explicitly.")
        if embed_dim is None:
            raise ValueError("embed_dim must be specified explicitly.")
        if isinstance(depths, int):
            depths = (depths,)
        else:
            depths = tuple(depths)
        if isinstance(num_heads, int):
            num_heads = (num_heads,)
        else:
            num_heads = tuple(num_heads)
        if len(num_heads) == 1 and len(depths) > 1:
            num_heads = num_heads * len(depths)
        if len(num_heads) != len(depths):
            raise ValueError(f"num_heads must have length {len(depths)}, got {len(num_heads)}")

        self.feat_dim = embed_dim // 3
        self.img_size = img_size

        self.core_12 = MaskGuidedCrossAttention(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            drop_path=drop_path,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint
        )

        self.core_32 = MaskGuidedCrossAttention(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            drop_path=drop_path,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint
        )
        
        self.fuser = GatedFusionProjection2CToC(self.feat_dim)

    def forward(self, x1, x2, x3, M1, M2, M3, x_size, return_attention=False):
        H, W = x_size
        B, L, C = x2.shape
        assert L == H * W, "x_size is inconsistent with the token length"
        assert C == self.feat_dim, \
            f"feature channels {C} do not match self.feat_dim {self.feat_dim}; check embed_dim=3*C"

        if return_attention:
            out_12, attn_12 = self.core_12(x2, x1, M2, M1, x_size, return_attention=True)
            out_32, attn_32 = self.core_32(x2, x3, M2, M3, x_size, return_attention=True)
        else:
            out_12 = self.core_12(x2, x1, M2, M1, x_size)  # (B, HW, C)
            out_32 = self.core_32(x2, x3, M2, M3, x_size)  # (B, HW, C)

        out_12_img = out_12.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        out_32_img = out_32.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        fused_img = self.fuser(out_12_img, out_32_img)  # (B, C, H, W)

        fused_out = fused_img.flatten(2).transpose(1, 2).contiguous()  # (B, HW, C)
        if return_attention:
            return fused_out, {
                "src1": attn_12,
                "src3": attn_32,
            }
        return fused_out
