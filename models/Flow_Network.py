import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

div_size = 16

def warp(img, flow):
    """
    img:  (B, C, H, W)
    flow: (B, 2, H, W)  (dx, dy) in pixel units
    """
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W, device=img.device, dtype=img.dtype).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H, device=img.device, dtype=img.dtype).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1)  # (B,2,H,W)

    flow_ = torch.cat([
        flow[:, 0:1] / ((W - 1.0) / 2.0),
        flow[:, 1:2] / ((H - 1.0) / 2.0)
    ], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)  # (B,H,W,2)

    return F.grid_sample(img, grid_, mode='bilinear', padding_mode='border', align_corners=True)

def resize(x, scale_factor):
    return F.interpolate(
        x, scale_factor=scale_factor, mode="bilinear",
        align_corners=False, recompute_scale_factor=True
    )

def convrelu(in_channels, out_channels, kernel_size=3, stride=1,
             padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride, padding, dilation, groups, bias=bias),
        nn.PReLU(out_channels)
    )

class DWSeparableConvPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=True):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=bias
        )
        self.act1 = nn.PReLU(in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.act2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.act1(self.dw(x))
        x = self.act2(self.pw(x))
        return x

def dwsep_convrelu(in_channels, out_channels, kernel_size=3, stride=1,
                   padding=1, dilation=1, bias=True):
    return DWSeparableConvPReLU(
        in_channels, out_channels,
        kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=dilation, bias=bias
    )

def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)

def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    if groups <= 1:
        return x
    assert c % groups == 0, f"channels={c} must be divisible by groups={groups}"
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, -1, h, w)
    return x

class Encoder(nn.Module):
    def __init__(self, in_channels=24, base_ch=40, use_dwsep=True):
        super().__init__()
        conv2 = dwsep_convrelu if use_dwsep else convrelu

        self.pyramid1 = nn.Sequential(
            convrelu(in_channels, base_ch, 3, 2, 1),
            conv2(base_ch, base_ch, 3, 1, 1),
        )
        self.pyramid2 = nn.Sequential(
            convrelu(base_ch, base_ch, 3, 2, 1),
            conv2(base_ch, base_ch, 3, 1, 1),
        )
        self.pyramid3 = nn.Sequential(
            convrelu(base_ch, base_ch, 3, 2, 1),
            conv2(base_ch, base_ch, 3, 1, 1),
        )
        self.pyramid4 = nn.Sequential(
            convrelu(base_ch, base_ch, 3, 2, 1),
            conv2(base_ch, base_ch, 3, 1, 1),
        )

    def forward(self, img_c):
        f1 = self.pyramid1(img_c)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f1, f2, f3, f4

class Decoder(nn.Module):
    def __init__(self, base_ch=40, dec_ch=120, shuffle_groups=3,
                 dwsep_conv1=True, dwsep_conv5=True, dwsep_mid=False):
        super().__init__()
        self.shuffle_groups = shuffle_groups

        in_ch = base_ch * 3 + 4

        def c3(in_c, out_c, groups=1, use_dw=False):
            if use_dw:
                return dwsep_convrelu(in_c, out_c, 3, 1, 1)
            return convrelu(in_c, out_c, 3, 1, 1, groups=groups)

        self.conv1 = c3(in_ch,  dec_ch, groups=1,             use_dw=dwsep_conv1)
        self.conv2 = c3(dec_ch, dec_ch, groups=shuffle_groups, use_dw=dwsep_mid)
        self.conv3 = c3(dec_ch, dec_ch, groups=shuffle_groups, use_dw=dwsep_mid)
        self.conv4 = c3(dec_ch, dec_ch, groups=shuffle_groups, use_dw=dwsep_mid)
        self.conv5 = c3(dec_ch, dec_ch, groups=1,              use_dw=dwsep_conv5)

        self.conv6 = deconv(dec_ch, 4)

    def forward(self, f0, f1, f2, flow0, flow2):
        f0_warp = warp(f0, flow0)
        f2_warp = warp(f2, flow2)
        f_in = torch.cat([f0_warp, f1, f2_warp, flow0, flow2], 1)

        f_out = self.conv1(f_in)
        f_out = channel_shuffle(self.conv2(f_out), self.shuffle_groups)
        f_out = channel_shuffle(self.conv3(f_out), self.shuffle_groups)
        f_out = channel_shuffle(self.conv4(f_out), self.shuffle_groups)
        f_out = self.conv5(f_out)
        f_out = self.conv6(f_out)

        up_flow0 = 2.0 * resize(flow0, scale_factor=2.0) + f_out[:, 0:2]
        up_flow2 = 2.0 * resize(flow2, scale_factor=2.0) + f_out[:, 2:4]
        return up_flow0, up_flow2

class FlowNetwork(nn.Module):
    def __init__(self, in_channels=24, base_ch=32, dec_ch=96, shuffle_groups=3,
                 enc_dwsep=True, dwsep_conv1=True, dwsep_conv5=True, dwsep_mid=False):
        super().__init__()
        if not dwsep_mid:
            assert dec_ch % shuffle_groups == 0, "dec_ch must be divisible by shuffle_groups"
        self.encoder = Encoder(in_channels=in_channels, base_ch=base_ch, use_dwsep=enc_dwsep)
        self.decoder = Decoder(
            base_ch=base_ch, dec_ch=dec_ch, shuffle_groups=shuffle_groups,
            dwsep_conv1=dwsep_conv1, dwsep_conv5=dwsep_conv5, dwsep_mid=dwsep_mid
        )

    def forward(self, img0_c, img1_c, img2_c, scale_factor=0.5):
        h, w = img1_c.shape[-2:]
        org_size = (int(h), int(w))
        input_size = (
            int(div_size * np.ceil(h * scale_factor / div_size)),
            int(div_size * np.ceil(w * scale_factor / div_size))
        )

        if input_size != org_size:
            img0_c = F.interpolate(img0_c, size=input_size, mode='bilinear', align_corners=False)
            img1_c = F.interpolate(img1_c, size=input_size, mode='bilinear', align_corners=False)
            img2_c = F.interpolate(img2_c, size=input_size, mode='bilinear', align_corners=False)

        f0_1, f0_2, f0_3, f0_4 = self.encoder(img0_c)
        f1_1, f1_2, f1_3, f1_4 = self.encoder(img1_c)
        f2_1, f2_2, f2_3, f2_4 = self.encoder(img2_c)

        up_flow0_5 = torch.zeros_like(f1_4[:, 0:2])
        up_flow2_5 = torch.zeros_like(f1_4[:, 0:2])

        up_flow0_4, up_flow2_4 = self.decoder(f0_4, f1_4, f2_4, up_flow0_5, up_flow2_5)
        up_flow0_3, up_flow2_3 = self.decoder(f0_3, f1_3, f2_3, up_flow0_4, up_flow2_4)
        up_flow0_2, up_flow2_2 = self.decoder(f0_2, f1_2, f2_2, up_flow0_3, up_flow2_3)
        up_flow0_1, up_flow2_1 = self.decoder(f0_1, f1_1, f2_1, up_flow0_2, up_flow2_2)

        if input_size != org_size:
            scale_h = org_size[0] / input_size[0]
            scale_w = org_size[1] / input_size[1]

            up_flow0_1 = F.interpolate(up_flow0_1, size=org_size, mode='bilinear', align_corners=False)
            up_flow0_1[:, 0] *= scale_w
            up_flow0_1[:, 1] *= scale_h

            up_flow2_1 = F.interpolate(up_flow2_1, size=org_size, mode='bilinear', align_corners=False)
            up_flow2_1[:, 0] *= scale_w
            up_flow2_1[:, 1] *= scale_h

        return up_flow0_1, up_flow2_1
