import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
try:
    from timm.layers import DropPath, to_2tuple, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def window_partition(x, window_size):
    # x: (B, H, W, C)
    B, H, W, C = x.shape
    x = x.view(B,
               H // window_size, window_size,
               W // window_size, window_size,
               C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous() \
               .view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    # windows: (B*nW, window, window, C) -> (B, H, W, C)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B,
                     H // window_size,
                     W // window_size,
                     window_size,
                     window_size,
                     -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous() \
         .view(B, H, W, -1)
    return x


########################################
# Window Cross Attention
########################################
class WindowCrossAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) *
                        (2 * window_size[1] - 1),
                        num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)      # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # Wh*Ww, Wh*Ww, 2
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        # x: (B_, N, C) -> K, V
        # y: (B_, N, C) -> Q
        B_, N, C = x.shape

        q = self.q(y).reshape(B_, N, 1, self.num_heads,
                              C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(x).reshape(B_, N, 2, self.num_heads,
                                C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q, kv[0], kv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (1, B_, num_heads, N, N)

        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)] \
            .view(self.window_size[0] * self.window_size[1],
                  self.window_size[0] * self.window_size[1],
                  -1)                       # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) \
                   + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


########################################
# Patch Embed / UnEmbed
########################################
class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4,
                 embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0],
                              img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # x: (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):

    def __init__(self, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        # x: (B, HW, C) -> (B, C, H, W)
        B, HW, C = x.shape
        H, W = x_size
        x = x.transpose(1, 2).view(B, self.embed_dim, H, W)
        return x

class SwinTransformerBlock_CrossAttention(nn.Module):

    def __init__(self, dim, input_resolution, num_heads,
                 window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.attn = WindowCrossAttention(
            dim, window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.patch_embed = PatchEmbed(embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)
        self.mlp = nn.Conv2d(dim, dim, 3, 1, 1,
                             groups=dim, padding_mode='reflect')

        self.register_buffer("attn_mask", None)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)) \
                             .masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, y, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = x + y
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            shifted_x = x
            shifted_y = y

        x_windows = window_partition(shifted_x, self.window_size)
        y_windows = window_partition(shifted_y, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if self.input_resolution == x_size and self.attn_mask is not None:
            attn_windows = self.attn(x_windows, y_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(
                x_windows,
                y_windows,
                mask=self.calculate_mask(x_size).to(x.device)
            )

        attn_windows = attn_windows.view(-1,
                                         self.window_size,
                                         self.window_size,
                                         C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x,
                           shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.patch_embed(self.mlp(self.patch_unembed(x, x_size)))
        return x

class CrossAttentionGroup(nn.Module):

    def __init__(self, dim, input_resolution, depth,
                 num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock_CrossAttention(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(input_resolution,
                                         dim=dim,
                                         norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, y, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, y, x_size)
            else:
                x = blk(x, y, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class SpatialAttentionAlignmentModule(nn.Module):

    def __init__(self, dim, input_resolution, depth,
                 num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False,
                 img_size=224, patch_size=4,
                 resi_connection='1conv'):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = CrossAttentionGroup(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint
        )

    def forward(self, x, y, x_size):
        return self.residual_group(x, y, x_size) + x


class SourceGuidedAttention(nn.Module):

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

        if embed_dim is None:
            raise ValueError("embed_dim must be specified explicitly.")
        if depths is None:
            raise ValueError("depths must be specified explicitly.")
        if num_heads is None:
            raise ValueError("num_heads must be specified explicitly.")
        if window_size is None:
            raise ValueError("window_size must be specified explicitly.")
        depths = tuple(depths)
        num_heads = tuple(num_heads)
        if len(num_heads) == 1 and len(depths) > 1:
            num_heads = num_heads * len(depths)
        if len(num_heads) != len(depths):
            raise ValueError(f"num_heads must have length {len(depths)}, got {len(num_heads)}")

        self.window_size = window_size
        self.num_layers = len(depths)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )

        if drop_path is not None:
            dpr_aligment = list(drop_path)
            assert len(dpr_aligment) == sum(depths), "drop_path length must equal sum(depths)"
        else:
            dpr_aligment = [x.item() for x in torch.linspace(
                0, drop_path_rate, sum(depths)
            )]

        self.a_12 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SpatialAttentionAlignmentModule(
                dim=embed_dim // 3,
                input_resolution=(self.patch_embed.patches_resolution[0],
                                  self.patch_embed.patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=1,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_aligment[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection='1conv'
            )
            self.a_12.append(layer)

    def forward(self, x, y, x_size):
        out = x
        for a_12 in self.a_12:
            out = a_12(out, y, x_size)
        return out
