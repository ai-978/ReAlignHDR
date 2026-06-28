import torch
import torch.nn as nn


class GatedFusionProjection2CToC(nn.Module):
    """
    Two-path Gated Fusion Projection (2C -> C), used inside SAB and TCB.

    Structure:
      cat(2C) -> gate(2C) -> gated(2C) -> proj(2C->C) -> 3x3 refine (residual)
    """

    def __init__(self, channels: int):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)

        mid = 2 * self.channels

        self.gate = nn.Sequential(
            nn.Conv2d(mid, mid, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(mid, mid, 1, 1, 0),
            nn.Sigmoid(),
        )

        self.proj = nn.Sequential(
            nn.Conv2d(mid, mid, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(mid, self.channels, 1, 1, 0),
        )

        self.refine = nn.Conv2d(self.channels, self.channels, 3, 1, 1, padding_mode="reflect")

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: a{tuple(a.shape)} vs b{tuple(b.shape)}")
        if a.dim() != 4:
            raise ValueError(f"expected 4D tensors (B,C,H,W), got {a.dim()}D")
        if a.shape[1] != self.channels:
            raise ValueError(f"expected C={self.channels}, got {a.shape[1]}")

        x = torch.cat([a, b], dim=1)          # (B, 2C, H, W)
        x = x * self.gate(x)                  # gated (B, 2C, H, W)
        y = self.proj(x)                      # (B, C, H, W)
        y = y + self.refine(y)                # residual refine
        return y
