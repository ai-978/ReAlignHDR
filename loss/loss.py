import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils.utils import *

try:
    from loss.vgg19 import VGG19_relu
    HAS_VGG = True
except ImportError:
    HAS_VGG = False


def warp(img, flow):
    """Warp img according to flow (backward warping)."""
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([
        flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
        flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)
    ], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(
        input=img,
        grid=grid_,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    return output


def range_compressor(hdr_img, mu=5000):
    """μ-law tone mapping."""
    if isinstance(hdr_img, np.ndarray):
        return (np.log(1 + mu * hdr_img)) / math.log(1 + mu)
    elif isinstance(hdr_img, torch.Tensor):
        return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)
    else:
        raise NotImplementedError(f'range compressor for [{type(hdr_img)}] is not found')


class L1Loss(nn.Module):
    """Standard L1 loss."""
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred, gt):
        return nn.L1Loss()(pred, gt)


class FFTLoss(nn.Module):
    """Frequency domain L1 loss."""
    def __init__(self, loss_weight=1.0, patch_size=0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss(reduction=reduction)
        self.ps = patch_size

    def forward(self, pred, target):
        if self.ps > 0:
            B, C, H, W = pred.size()
            grid_height, grid_width = H // self.ps, W // self.ps

            pred_patch = rearrange(
                pred, "n c (gh bh) (gw bw) -> n (c gh gw) bh bw",
                gh=grid_height, gw=grid_width, bh=self.ps, bw=self.ps
            )
            target_patch = rearrange(
                target, "n c (gh bh) (gw bw) -> n (c gh gw) bh bw",
                gh=grid_height, gw=grid_width, bh=self.ps, bw=self.ps
            )

            pred_fft = torch.fft.rfft2(pred_patch, dim=(-2, -1))
            target_fft = torch.fft.rfft2(target_patch, dim=(-2, -1))

            pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
            target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        else:
            pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
            target_fft = torch.fft.rfft2(target, dim=(-2, -1))

            pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
            target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)


class PerceptualLoss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.criterion = nn.L1Loss()
        if HAS_VGG:
            self.vgg = VGG19_relu()
            self.vgg.eval()
            for p in self.vgg.parameters():
                p.requires_grad = False
        else:
            self.vgg = None
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        self.weights = weights

    def forward(self, x, y):
        if self.vgg is None:
            return torch.tensor(0.0, device=x.device)
        
        x = (x - self.mean.to(x)) / self.std.to(x)
        y = (y - self.mean.to(y)) / self.std.to(y)
        
        x_vgg = self.vgg(x, None)
        y_vgg = self.vgg(y, None)

        loss = 0.0
        loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])
        return loss


class HDRLoss(nn.Module):
    def __init__(self, lambda_fft=None, lambda_perc=None, mu=5000):
        super(HDRLoss, self).__init__()
        if lambda_fft is None:
            raise ValueError("lambda_fft must be specified explicitly.")
        if lambda_perc is None:
            raise ValueError("lambda_perc must be specified explicitly.")
        self.mu = mu
        self.lambda_fft = lambda_fft
        self.lambda_perc = lambda_perc
        self.loss_recon = L1Loss()
        if self.lambda_fft > 0:
            self.loss_le = FFTLoss()
        else:
            self.loss_le = None
        if self.lambda_perc > 0 and HAS_VGG:
            self.loss_perc = PerceptualLoss()
        else:
            self.loss_perc = None

    def forward(self, pred, gt):
        """
        pred, gt: [B,3,H,W] HDR
        """
        loss = 0.0
        loss_dict = {}

        pred_mu = range_compressor(pred, self.mu)
        gt_mu = range_compressor(gt, self.mu)

        loss_recon = self.loss_recon(pred_mu, gt_mu)
        loss_dict['loss_recon'] = loss_recon
        loss = loss + loss_recon

        if self.loss_le is not None:
            loss_le = self.loss_le(pred_mu, gt_mu) * self.lambda_fft
            loss_dict['loss_le'] = loss_le
            loss = loss + loss_le

        if self.loss_perc is not None:
            loss_perc = self.loss_perc(pred_mu, gt_mu) * self.lambda_perc
            loss_dict['loss_perc'] = loss_perc
            loss = loss + loss_perc

        loss_dict['loss_hdr'] = loss
        return loss, loss_dict


# ----------------- Mask Shape Loss & Census Loss -----------------

def prepare_masks(mask1, mask2, mask3):
    if mask1.dim() == 3:
        mask1 = mask1.unsqueeze(1)
        mask2 = mask2.unsqueeze(1)
        mask3 = mask3.unsqueeze(1)
    return mask1.float(), mask2.float(), mask3.float()


def rgb_to_gray(x):
    if x.shape[1] == 1:
        return x
    if x.shape[1] >= 6:
        r, g, b = x[:, 3:4], x[:, 4:5], x[:, 5:6]
    else:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def ternary_transform(img, patch_size=7):
    B, C, H, W = img.shape
    assert C == 1
    pad = patch_size // 2
    img_pad = F.pad(img, (pad, pad, pad, pad), mode='replicate')
    patches = F.unfold(img_pad, kernel_size=patch_size)  # [B,P,H*W]
    patches = patches.view(B, patch_size * patch_size, H, W)
    center = img.repeat(1, patch_size * patch_size, 1, 1)
    diff = patches - center
    eps = 0.03
    diff_norm = torch.sqrt(diff ** 2 + eps ** 2)
    t = diff / diff_norm
    return t


def census_distance(t1, t2):
    dist = (t1 - t2).abs()
    dist = dist.mean(1, keepdim=True)
    return dist


class CensusLoss(nn.Module):
    def __init__(self, patch_size=7):
        super(CensusLoss, self).__init__()
        self.patch_size = patch_size

    def forward(self, I1, I2, I3, flow_1_2, flow_3_2):
        B, _, Hf, Wf = flow_1_2.shape

        I1_hdr = I1[:, 0:3, :, :] if I1.shape[1] >= 6 else I1[:, 0:3, :, :]
        I2_hdr = I2[:, 0:3, :, :] if I2.shape[1] >= 6 else I2[:, 0:3, :, :]
        I3_hdr = I3[:, 0:3, :, :] if I3.shape[1] >= 6 else I3[:, 0:3, :, :]
        
        I1_mu = range_compressor(I1_hdr)
        I2_mu = range_compressor(I2_hdr)
        I3_mu = range_compressor(I3_hdr)

        I1_f = F.interpolate(I1_mu, size=(Hf, Wf), mode='bilinear', align_corners=False)
        I2_f = F.interpolate(I2_mu, size=(Hf, Wf), mode='bilinear', align_corners=False)
        I3_f = F.interpolate(I3_mu, size=(Hf, Wf), mode='bilinear', align_corners=False)

        I1_warp = warp(I1_f, flow_1_2)
        I3_warp = warp(I3_f, flow_3_2)

        I2_gray = 0.2989 * I2_f[:, 0:1] + 0.5870 * I2_f[:, 1:2] + 0.1140 * I2_f[:, 2:3]
        I1_warp_gray = 0.2989 * I1_warp[:, 0:1] + 0.5870 * I1_warp[:, 1:2] + 0.1140 * I1_warp[:, 2:3]
        I3_warp_gray = 0.2989 * I3_warp[:, 0:1] + 0.5870 * I3_warp[:, 1:2] + 0.1140 * I3_warp[:, 2:3]

        # Census transform
        t2 = ternary_transform(I2_gray, patch_size=self.patch_size)
        t1_warp = ternary_transform(I1_warp_gray, patch_size=self.patch_size)
        t3_warp = ternary_transform(I3_warp_gray, patch_size=self.patch_size)

        # Census distance
        dist1 = census_distance(t2, t1_warp)
        dist3 = census_distance(t2, t3_warp)

        loss_census = dist1.mean() + dist3.mean()

        return loss_census, {'loss_census': loss_census}


class MaskShapeLoss(nn.Module):
    def __init__(self):
        super(MaskShapeLoss, self).__init__()

    def forward(self, flow_1_2, flow_3_2, mask1, mask2, mask3):
        mask1, mask2, mask3 = prepare_masks(mask1, mask2, mask3)
        mask1 = (mask1 > 0).float()
        mask2 = (mask2 > 0).float()
        mask3 = (mask3 > 0).float()

        B, _, Hf, Wf = flow_1_2.shape

        mask1_f = F.interpolate(mask1, size=(Hf, Wf), mode='nearest')
        mask2_f = F.interpolate(mask2, size=(Hf, Wf), mode='nearest')
        mask3_f = F.interpolate(mask3, size=(Hf, Wf), mode='nearest')

        # warp mask
        mask1_warp = warp(mask1_f, flow_1_2)
        mask3_warp = warp(mask3_f, flow_3_2)

        diff1 = (mask1_warp - mask2_f).abs()
        diff3 = (mask3_warp - mask2_f).abs()

        loss_shape = diff1.mean() + diff3.mean()

        return loss_shape, {'loss_shape': loss_shape}


class loss_total(nn.Module):
    """
    Weight values are intentionally not hard-coded here. Pass them from the
    training script so the experiment configuration matches the paper or an
    explicit ablation setting.
    """
    def __init__(self,
                 lambda_fft=None,
                 lambda_perc=None,
                 mu=5000,
                 lambda_shape=0,
                 lambda_census=None,
                 census_patch_size=7):
        super(loss_total, self).__init__()
        if lambda_census is None:
            raise ValueError("lambda_census must be specified explicitly.")
        self.hdr_loss = HDRLoss(lambda_fft=lambda_fft, lambda_perc=lambda_perc, mu=mu)
        self.shape_loss = MaskShapeLoss()
        self.census_loss = CensusLoss(patch_size=census_patch_size)
        self.lambda_shape = lambda_shape
        self.lambda_census = lambda_census

    def forward(self,
                pred, label,
                I1, I2, I3,
                mask1, mask2, mask3,  # SAM mask
                flow_1_2, flow_3_2):
        """
        pred:      [B,3,H,W] HDR prediction
        label:     [B,3,H,W] HDR GT
        I1,I2,I3:  [B,C,H,W] LDR1/2/3
        mask1/2/3: [B,H,W] or [B,1,H,W]
        flow_*_*:  [B,2,Hf,Wf]
        """

        loss_hdr, hdr_dict = self.hdr_loss(pred, label)

        loss_shape, shape_dict = self.shape_loss(flow_1_2, flow_3_2,
                                                  mask1, mask2, mask3)

        loss_census, census_dict = self.census_loss(I1, I2, I3,
                                                     flow_1_2, flow_3_2)

        loss = loss_hdr + self.lambda_shape * loss_shape + self.lambda_census * loss_census

        loss_dict = {
            'loss_total': loss,
            'loss_hdr': loss_hdr,
            'loss_shape': loss_shape,
            'loss_census': loss_census,
        }
        loss_dict.update(hdr_dict)

        return loss, loss_dict
