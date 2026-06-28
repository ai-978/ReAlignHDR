import torch
from torch import nn as nn
from einops import rearrange
try:
    from timm.layers import DropPath
except ImportError:
    from timm.models.layers import DropPath


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


class Mutual_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        assert x.shape == y.shape
        b, c, h, w = x.shape

        q = self.q(y)
        k = self.k(x)
        v = self.v(x)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out


class CrossAttentionProximalOperater(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()

        self.attn = Mutual_Attention(dim, num_heads, bias)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1,
                              groups=dim, padding_mode='reflect')

    def forward(self, x, x_size):
        b, n, c = x.shape
        h, w = x_size
        assert n == h * w

        x = x.permute(0, 2, 1).view(b, c, h, w)
        y = x

        x = x + self.attn(x, y)
        x = x + self.conv(x)

        x = to_3d(x)

        return x


class Channel_Self_Attention(nn.Module):
    def __init__(self,
                 embed_dim=None,
                 depths=None,
                 num_heads=None,
                 drop_path_rate=0.,
                 drop_path=None):
        super().__init__()

        if embed_dim is None:
            raise ValueError("embed_dim must be specified explicitly.")
        if depths is None:
            raise ValueError("depths must be specified explicitly.")
        if num_heads is None:
            raise ValueError("num_heads must be specified explicitly.")
        self.depths = tuple(depths)
        num_heads = tuple(num_heads)
        if len(num_heads) == 1 and len(self.depths) > 1:
            num_heads = num_heads * len(self.depths)
        if len(num_heads) != len(self.depths):
            raise ValueError(f"num_heads must have length {len(self.depths)}, got {len(num_heads)}")
        total_depth = sum(self.depths)
        if drop_path is not None:
            dpr = list(drop_path)
            assert len(dpr) == total_depth, f"drop_path length must be {total_depth}"
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.f_12 = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        dp_idx = 0
        for stage_idx, depth in enumerate(self.depths):
            heads = num_heads[stage_idx] if stage_idx < len(num_heads) else num_heads[-1]
            for _ in range(depth):
                self.f_12.append(CrossAttentionProximalOperater(
                    dim=embed_dim // 3,
                    num_heads=heads,
                    bias=False
                ))
                rate = dpr[dp_idx]
                self.drop_paths.append(DropPath(rate) if rate > 0 else nn.Identity())
                dp_idx += 1

    def forward(self, x, x_size):
        out = x
        for f_12, drop_path in zip(self.f_12, self.drop_paths):
            delta = f_12(out, x_size) - out
            out = out + drop_path(delta)
        return out
