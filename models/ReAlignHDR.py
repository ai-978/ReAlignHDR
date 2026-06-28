import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_

try:
    from .Texture_Compensation_Branch import TextureCompensationBranch
    from .Structure_Anchoring_Branch import StructureAnchoringBranch
    from .Spatial_Self_Attention import Spatial_Self_Attention
    from .Channel_Cross_Attention import Channel_Self_Attention
except ImportError:
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from Texture_Compensation_Branch import TextureCompensationBranch
    from Structure_Anchoring_Branch import StructureAnchoringBranch
    from Spatial_Self_Attention import Spatial_Self_Attention
    from Channel_Cross_Attention import Channel_Self_Attention


def _make_dpr(depths, drop_path_rate, drop_path=None):
    total_depth = sum(depths)
    if drop_path is not None:
        dpr = list(drop_path)
        assert len(dpr) == total_depth, f"drop_path length must be {total_depth}"
        return dpr
    return torch.linspace(0, drop_path_rate, steps=total_depth).tolist()


def _as_tuple(value, name):
    if value is None:
        raise ValueError(f"{name} must be specified explicitly.")
    if isinstance(value, int):
        return (value,)
    return tuple(int(v) for v in value)


def _match_tuple(value, name, length):
    values = _as_tuple(value, name)
    if len(values) == 1 and length > 1:
        return values * length
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}, got {len(values)}")
    return values


def _require(value, name):
    if value is None:
        raise ValueError(f"{name} must be specified explicitly.")
    return value

class ProgressiveRefinementModule(nn.Module):
    def __init__(self, base_dim, img_size=(128, 128),
                 stage_blocks=None,
                 stage_heads=None,
                 window_size=None,
                 mlp_ratio=2.,
                 drop_rate=0.,
                 drop_path_rate=None):
        super(ProgressiveRefinementModule, self).__init__()

        self.base_dim = base_dim
        self.dim1 = base_dim * 3
        self.dim2 = base_dim * 2
        self.dim3 = base_dim * 1
        stage_blocks = _match_tuple(stage_blocks, "prm_stage_blocks", 3)
        stage_heads = _match_tuple(stage_heads, "prm_stage_heads", 3)
        window_size = _require(window_size, "window_size")
        drop_path_rate = _require(drop_path_rate, "drop_path_rate")

        if len(stage_blocks) != 3:
            raise ValueError("stage_blocks must have length 3")
        if len(stage_heads) != 3:
            raise ValueError("stage_heads must have length 3")
        if any(int(b) <= 0 for b in stage_blocks):
            raise ValueError("stage_blocks values must be positive integers")

        self.stage_blocks = tuple(int(b) for b in stage_blocks)
        self.stage_heads = tuple(int(h) for h in stage_heads)

        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate

        if isinstance(img_size, int):
            self.h = self.w = img_size
        else:
            self.h, self.w = img_size

        total_blocks = sum(self.stage_blocks)
        if total_blocks == 1:
            dpr_all = [float(self.drop_path_rate)]
        else:
            dpr_all = torch.linspace(0, self.drop_path_rate, steps=total_blocks).tolist()

        b1, b2, b3 = self.stage_blocks
        h1, h2, h3 = self.stage_heads

        if self.dim1 % h1 != 0:
            raise ValueError(f"stage1: dim={self.dim1} must be divisible by heads={h1}")
        if self.dim2 % h2 != 0:
            raise ValueError(f"stage2: dim={self.dim2} must be divisible by heads={h2}")
        if self.dim3 % h3 != 0:
            raise ValueError(f"stage3: dim={self.dim3} must be divisible by heads={h3}")

        dpr_1 = dpr_all[0:b1]
        dpr_2 = dpr_all[b1:b1 + b2]
        dpr_3 = dpr_all[b1 + b2:b1 + b2 + b3]

        # -------------------------
        # stage1: 3C -> 2C
        # -------------------------
        depths_1 = (b1,)
        self.spatial_1 = Spatial_Self_Attention(
            img_size=(self.h, self.w),
            patch_size=1,
            embed_dim=self.dim1,
            depths=depths_1,
            num_heads=(h1,),
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=self.drop_rate,
            attn_drop_rate=0.,
            drop_path=dpr_1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            resi_connection='1conv'
        )
        self.channel_1 = Channel_Self_Attention(
            embed_dim=self.dim1 * 3,
            depths=depths_1,
            num_heads=(h1,),
            drop_path=dpr_1
        )
        self.reduce_1 = self._GatedReduce(self.dim1, self.dim2)

        # -------------------------
        # stage2: 2C -> C
        # -------------------------
        depths_2 = (b2,)
        self.spatial_2 = Spatial_Self_Attention(
            img_size=(self.h, self.w),
            patch_size=1,
            embed_dim=self.dim2,
            depths=depths_2,
            num_heads=(h2,),
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=self.drop_rate,
            attn_drop_rate=0.,
            drop_path=dpr_2,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            resi_connection='1conv'
        )
        
        self.channel_2 = Channel_Self_Attention(
            embed_dim=self.dim2 * 3,
            depths=depths_2,
            num_heads=(h2,),
            drop_path=dpr_2
        )
        self.reduce_2 = self._GatedReduce(self.dim2, self.dim3)

        # -------------------------
        # stage3: C -> C
        # -------------------------
        depths_3 = (b3,)
        self.spatial_3 = Spatial_Self_Attention(
            img_size=(self.h, self.w),
            patch_size=1,
            embed_dim=self.dim3,
            depths=depths_3,
            num_heads=(h3,),
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=self.drop_rate,
            attn_drop_rate=0.,
            drop_path=dpr_3,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            resi_connection='1conv'
        )
        
        self.channel_3 = Channel_Self_Attention(
            embed_dim=self.dim3 * 3,
            depths=depths_3,
            num_heads=(h3,),
            drop_path=dpr_3
        )

    class _GatedReduce(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.in_ch = int(in_ch)
            self.out_ch = int(out_ch)

            self.gate = nn.Sequential(
                nn.Conv2d(self.in_ch, self.in_ch, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.in_ch, self.in_ch, 1, 1, 0),
                nn.Sigmoid(),
            )

            hidden = max(self.out_ch * 2, self.in_ch // 2, 16)
            self.proj = nn.Sequential(
                nn.Conv2d(self.in_ch, hidden, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(hidden, self.out_ch, 1, 1, 0),
            )

            self.skip = nn.Conv2d(self.in_ch, self.out_ch, 1, 1, 0)
            self.refine = nn.Conv2d(self.out_ch, self.out_ch, 3, 1, 1, padding_mode="reflect")

        def forward(self, x):
            x_g = x * self.gate(x)
            y = self.proj(x_g) + self.skip(x)
            y = y + self.refine(y)
            return y

    def _run_stage(self, x, x_size, spatial, channel, reduce=None):
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)              # (B, N, C)
        x_flat = spatial(x_flat, x_size)                   # (B, N, C)

        x_flat = channel(x_flat, x_size)                   # (B, N, C)

        x = x_flat.transpose(1, 2).view(B, C, H, W).contiguous()

        if reduce is not None:
            x = reduce(x)
        return x

    def forward(self, tex_feat, x2, art_feat):
        x = torch.cat([tex_feat, x2, art_feat], dim=1)
        x_size = (x.shape[2], x.shape[3])

        # 3C -> 2C -> C
        x = self._run_stage(x, x_size, self.spatial_1, self.channel_1, self.reduce_1)
        x = self._run_stage(x, x_size, self.spatial_2, self.channel_2, self.reduce_2)
        x = self._run_stage(x, x_size, self.spatial_3, self.channel_3, reduce=None)

        return x

class ReAlignHDR(nn.Module):
    """
    ReAlignHDR architecture:
      - TextureCompensationBranch transports aligned textures.
      - StructureAnchoringBranch enforces region-constrained interaction.
      - ProgressiveRefinementModule integrates SAB, reference, and TCB features.
    """

    def __init__(self, embed_dim=None, img_size=(128, 128),
                 tcb_depths=None,
                 sab_depths=None,
                 tcb_num_heads=None,
                 sab_num_heads=None,
                 prm_stage_blocks=None,
                 prm_stage_heads=None,
                 window_size=None,
                 num_mlp_layers=1,
                 mlp_ratio=2.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 fusion_drop_path_rate=0.):
        super(ReAlignHDR, self).__init__()

        embed_dim = _require(embed_dim, "embed_dim")
        window_size = _require(window_size, "window_size")
        tcb_depths = _as_tuple(tcb_depths, "tcb_depths")
        sab_depths = _as_tuple(sab_depths, "sab_depths")
        tcb_num_heads = _match_tuple(tcb_num_heads, "tcb_num_heads", len(tcb_depths))
        sab_num_heads = _match_tuple(sab_num_heads, "sab_num_heads", len(sab_depths))
        prm_stage_blocks = _match_tuple(prm_stage_blocks, "prm_stage_blocks", 3)
        prm_stage_heads = _match_tuple(prm_stage_heads, "prm_stage_heads", 3)

        assert embed_dim % 3 == 0, "embed_dim must be divisible by 3"
        self.total_dim = embed_dim
        self.base_dim = embed_dim // 3

        dpr_tex = _make_dpr(tcb_depths, drop_path_rate)
        dpr_art = _make_dpr(sab_depths, drop_path_rate)
        if isinstance(img_size, int):
            self.h = self.w = img_size
        else:
            self.h, self.w = img_size

        self.num_mlp_layers = num_mlp_layers
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate

        self.window_size = window_size

        self.shallow_conv1 = nn.Conv2d(6, self.base_dim, 3, 1, 1, padding_mode='reflect')
        self.shallow_conv2 = nn.Conv2d(6, self.base_dim, 3, 1, 1, padding_mode='reflect')
        self.shallow_conv3 = nn.Conv2d(6, self.base_dim, 3, 1, 1, padding_mode='reflect')

        # Texture Compensation Branch (TCB): flow + source-guided attention
        self.tcb = TextureCompensationBranch(
            feature_dim=self.base_dim,
            img_size=img_size,
            depth=tcb_depths,
            num_heads=tcb_num_heads,
            window_size=self.window_size,
            drop_path_rate=drop_path_rate,
            drop_path=dpr_tex
        )

        # Structure Anchoring Branch (SAB): mask-guided cross-attention
        self.sab = StructureAnchoringBranch(
            img_size=(self.h, self.w),
            patch_size=1,
            embed_dim=self.total_dim,
            depths=sab_depths,
            num_heads=sab_num_heads,
            window_size=self.window_size,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=drop_path_rate,
            drop_path=dpr_art,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False
        )

        self.prm = ProgressiveRefinementModule(
            base_dim=self.base_dim,
            img_size=img_size,
            stage_blocks=prm_stage_blocks,
            stage_heads=prm_stage_heads,
            window_size=self.window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=0.,
            drop_path_rate=fusion_drop_path_rate,
        )

        self.conv_skip = nn.Conv2d(self.base_dim, self.base_dim, 3, 1, 1, padding_mode='reflect')
        self.out = nn.Sequential(
            nn.Conv2d(self.base_dim, 3, 3, 1, 1, padding_mode='reflect'),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _pad_to_window(self, L1, L2, L3, M1, M2, M3):
        _, _, H, W = L1.shape
        mod_pad_h = (self.window_size - H % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - W % self.window_size) % self.window_size
        if mod_pad_h == 0 and mod_pad_w == 0:
            return (L1, L2, L3, M1, M2, M3), (0, 0)
        pad_img = (0, mod_pad_w, 0, mod_pad_h)
        pad_mask = (0, mod_pad_w, 0, mod_pad_h)
        L1 = F.pad(L1, pad_img, mode='reflect')
        L2 = F.pad(L2, pad_img, mode='reflect')
        L3 = F.pad(L3, pad_img, mode='reflect')
        M1 = F.pad(M1, pad_mask, mode='replicate')
        M2 = F.pad(M2, pad_mask, mode='replicate')
        M3 = F.pad(M3, pad_mask, mode='replicate')
        return (L1, L2, L3, M1, M2, M3), (mod_pad_h, mod_pad_w)

    def forward(self, input, M1, M2, M3, return_features=False):

        LDR1, LDR2, LDR3 = input.chunk(3, dim=1)
        B, _, H_orig, W_orig = LDR1.shape

        (LDR1, LDR2, LDR3, M1, M2, M3), (pad_h, pad_w) = self._pad_to_window(
            LDR1, LDR2, LDR3, M1, M2, M3
        )
        x_size = (LDR1.shape[2], LDR1.shape[3])

        x1 = self.shallow_conv1(LDR1)
        x2 = self.shallow_conv2(LDR2)
        x3 = self.shallow_conv3(LDR3)

        tex_flat, flow_1_2, flow_3_2 = self.tcb(
            x1, x2, x3, LDR1, LDR2, LDR3, M1, M2, M3
        )
        tex_feat = tex_flat.transpose(1, 2).view(B, self.base_dim, *x_size).contiguous()

        x1_flat = x1.flatten(2).transpose(1, 2)
        x2_flat = x2.flatten(2).transpose(1, 2)
        x3_flat = x3.flatten(2).transpose(1, 2)
        if return_features:
            sab_flat, sab_attention = self.sab(
                x1_flat, x2_flat, x3_flat, M1, M2, M3, x_size, return_attention=True
            )
        else:
            sab_flat = self.sab(x1_flat, x2_flat, x3_flat, M1, M2, M3, x_size)
        sab_feat = sab_flat.transpose(1, 2).view(B, self.base_dim, *x_size).contiguous()

        fused_feat = self.prm(tex_feat, x2, sab_feat)

        out = fused_feat + self.conv_skip(x2)
        out = self.out(out)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H_orig, :W_orig]
            flow_1_2 = flow_1_2[:, :, :H_orig, :W_orig]
            flow_3_2 = flow_3_2[:, :, :H_orig, :W_orig]

        if return_features:
            aux = {
                "features": {
                    "SAB": sab_feat,
                    "TCB": tex_feat,
                    "fuse": fused_feat,
                },
                "attention": sab_attention,
            }
            return out, flow_1_2, flow_3_2, aux

        return out, flow_1_2, flow_3_2
