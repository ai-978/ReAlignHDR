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


def _masked_softmax(logits, mask, dim=-1, eps=1e-9):
    max_logits, _ = torch.max(logits, dim=dim, keepdim=True)   # (B, heads, N, 1)
    max_logits = torch.where(torch.isinf(max_logits),
                             torch.zeros_like(max_logits), max_logits)
    exps = torch.exp(logits - max_logits) * mask
    denom = exps.sum(dim=dim, keepdim=True) + eps
    return exps / denom


class WindowCrossAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.q  = nn.Linear(dim, dim,     bias=qkv_bias)   # from x (reference)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)   # from y (non-ref)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, y, sem_mask, mask=None, x_raw=None, return_attention=False):
        B_, N, C = x.shape
        assert y.shape == x.shape
        if x_raw is None:
            x_raw = x

        # Q from x, KV from y
        q = self.q(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B_,h,N,d
        kv = self.kv(y).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # B_,h,N,d

        q = q * self.scale
        logits = (q @ k.transpose(-2, -1))  # (B_, h, N, N)

        # relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)] \
            .view(self.window_size[0] * self.window_size[1],
                  self.window_size[0] * self.window_size[1],
                  -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # h, N, N
        logits = logits + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            logits = logits.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            logits = logits.view(-1, self.num_heads, N, N)

        allow = sem_mask.bool().unsqueeze(1)  # (B_,1,N,N)
        logits = logits.masked_fill(~allow, -1e4)

        attn = torch.softmax(logits, dim=-1)
        attn = attn * allow.float()
        den = attn.sum(dim=-1, keepdim=True)  # (B_,h,N,1)
        attn = torch.where(den > 0, attn / den, torch.zeros_like(attn))
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # delta
        out = self.proj(out)
        out = self.proj_drop(out)
        if return_attention:
            return out, attn
        return out

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
        self.mlp = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, padding_mode='reflect')

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
        return attn_mask  # (nW, N, N)

    @staticmethod
    def _reshape_mask(M, x_size, device):
        H, W = x_size
        B = M.shape[0]
        if M.dim() == 2:          # B, HW
            M = M.view(B, H, W, 1)
        elif M.dim() == 3:        # B, H, W
            M = M.unsqueeze(-1)
        elif M.dim() == 4:
            if M.shape[1] != 1:
                M = M[:, :1, ...]
            M = M.permute(0, 2, 3, 1)
        else:
            raise ValueError(f"Unsupported mask dim {M.dim()}")
        M = (M > 0.5).to(device)  # bool
        return M  # (B,H,W,1) bool

    def forward(self, x, y, Mx, My, x_size, return_attention=False):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "x_size is inconsistent with the token length"
        assert x.shape == y.shape

        shortcut = x

        # token -> image
        x_img = x.view(B, H, W, C)
        y_img = y.view(B, H, W, C)

        # mask -> image (bool)
        Mx_img = self._reshape_mask(Mx, x_size, x_img.device)
        My_img = self._reshape_mask(My, x_size, y_img.device)

        # shift
        if self.shift_size > 0:
            shifted_x  = torch.roll(x_img,  shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y  = torch.roll(y_img,  shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_Mx = torch.roll(Mx_img, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_My = torch.roll(My_img, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x, shifted_y   = x_img, y_img
            shifted_Mx, shifted_My = Mx_img, My_img

        # window partition
        N = self.window_size * self.window_size
        x_windows  = window_partition(shifted_x,  self.window_size).view(-1, N, C)
        y_windows  = window_partition(shifted_y,  self.window_size).view(-1, N, C)
        Mx_windows = window_partition(shifted_Mx, self.window_size).view(-1, N)  # bool
        My_windows = window_partition(shifted_My, self.window_size).view(-1, N)  # bool

        sem_mask = (Mx_windows.unsqueeze(2) == My_windows.unsqueeze(1))  # (B*nW, N, N) bool

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(x_size).to(x_img.device)
        else:
            attn_mask = None

        attn_out = self.attn(
            x=x_windows,
            y=y_windows,
            sem_mask=sem_mask,
            mask=attn_mask,
            x_raw=x_windows,
            return_attention=return_attention
        )
        if return_attention:
            delta_windows, attn = attn_out
        else:
            delta_windows = attn_out

        # windows -> image
        delta_windows = delta_windows.view(-1, self.window_size, self.window_size, C)
        shifted_delta = window_reverse(delta_windows, self.window_size, H, W)

        # reverse shift
        if self.shift_size > 0:
            delta_img = torch.roll(shifted_delta, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            delta_img = shifted_delta

        delta = delta_img.view(B, H * W, C)

        x_out = shortcut + self.drop_path(delta)
        x_out = x_out + self.patch_embed(self.mlp(self.patch_unembed(x_out, x_size)))
        if return_attention:
            return x_out, {
                "attention": attn,
                "gate": sem_mask.unsqueeze(1),
            }
        return x_out

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

    def forward(self, x, y, Mx, My, x_size, return_attention=False):
        records = [] if return_attention else None
        for blk in self.blocks:
            if self.use_checkpoint:
                if return_attention:
                    x, record = blk(x, y, Mx, My, x_size, return_attention=True)
                    records.append(record)
                else:
                    x = checkpoint.checkpoint(blk, x, y, Mx, My, x_size)
            else:
                if return_attention:
                    x, record = blk(x, y, Mx, My, x_size, return_attention=True)
                    records.append(record)
                else:
                    x = blk(x, y, Mx, My, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        if return_attention:
            return x, records
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

    def forward(self, x, y, Mx, My, x_size, return_attention=False):
        if return_attention:
            out, records = self.residual_group(x, y, Mx, My, x_size, return_attention=True)
            return out + x, records
        return self.residual_group(x, y, Mx, My, x_size) + x


class MaskGuidedCrossAttention(nn.Module):

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

    def forward(self, x, y, Mx, My, x_size, return_attention=False):
        out = x
        records = [] if return_attention else None
        for a_12 in self.a_12:
            if return_attention:
                out, layer_records = a_12(out, y, Mx, My, x_size, return_attention=True)
                records.extend(layer_records)
            else:
                out = a_12(out, y, Mx, My, x_size)
        if return_attention:
            return out, records
        return out
