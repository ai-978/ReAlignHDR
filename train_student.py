# -*- coding:utf-8 -*-
import argparse
import logging
import math
import os
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_ssim

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from lr_scheduler.mylr import MyLR
from models.ReAlignHDR import ReAlignHDR
from utils.pu21 import pu21_metrics


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def range_compressor(hdr_img, mu=5000):
    return torch.log(1 + mu * hdr_img) / math.log(1 + mu)


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_int_tuple(value):
    if isinstance(value, tuple):
        return value
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def pad_image(image, masks, patch_size, stride_size):
    _, _, h, w = image.size()
    pad_h = (stride_size - (h - patch_size) % stride_size) % stride_size
    pad_w = (stride_size - (w - patch_size) % stride_size) % stride_size
    padding = (0, pad_w, 0, pad_h)
    padded_image = F.pad(image, padding, mode="reflect")
    masks_bhw = masks.view(masks.shape[0] * masks.shape[1], 1, masks.shape[-2], masks.shape[-1])
    masks_bhw = F.pad(masks_bhw, padding, mode="replicate")
    padded_masks = masks_bhw.view(
        masks.shape[0], masks.shape[1], 1, masks_bhw.shape[-2], masks_bhw.shape[-1]
    )
    return padded_image, padded_masks


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return float(values.mean()), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def get_args():
    parser = argparse.ArgumentParser(
        description="Train mask-free ReAlignHDR student with privileged interaction distillation",
    )

    parser.add_argument("--teacher_ckpt", type=str, required=True, help="converged teacher checkpoint")
    parser.add_argument("--student_resume", type=str, help="resume a student checkpoint")
    parser.add_argument("--init_student_from_teacher", action="store_true",
                        help="initialize the student from the teacher checkpoint")
    parser.add_argument("--no_init_student_from_teacher", dest="init_student_from_teacher",
                        action="store_false")

    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--exposure_file_name", type=str, required=True)
    parser.add_argument("--ldr_prefix", type=str)
    parser.add_argument("--ldr_folder_name", type=str)
    parser.add_argument("--label_file_name", type=str, required=True)
    parser.add_argument("--mask_npy_name", type=str)

    parser.add_argument("--train_patch_size", type=int, required=True)
    parser.add_argument("--patch_size", type=int, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--test_batch_size", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--test_num_workers", type=int, required=True)
    parser.add_argument("--cache_choice", type=int, required=True,
                        help="cache for dataloader: 0 none, 1 bin, 2 in_memory")
    parser.add_argument("--foreground_patch_prob", type=float, required=True,
                        help="target probability of sampling a patch containing foreground mask pixels")
    parser.add_argument("--foreground_min_pixels", type=int, required=True,
                        help="minimum foreground pixels required for a foreground patch")
    parser.add_argument("--foreground_sample_attempts", type=int, required=True,
                        help="maximum crop attempts for foreground/background balanced sampling")

    parser.add_argument("--start_epoch", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--phase1_epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--lr_decay", action="store_true")
    parser.add_argument("--eta_min", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--log_interval", type=int, required=True)
    parser.add_argument("--eval_interval", type=int, required=True)
    parser.add_argument("--save_interval", type=int, required=True)
    parser.add_argument("--pu_peak_luminance", type=float, required=True,
                        help="display peak luminance in nits for PU21-absolute metrics")

    parser.add_argument("--lambda_fft", type=float, required=True,
                        help="weight of the FFT loss")
    parser.add_argument("--lambda_perc", type=float, required=True,
                        help="weight of the perceptual loss")
    parser.add_argument("--lambda_census", type=float, required=True,
                        help="weight of the census flow loss")
    parser.add_argument("--lambda_shape", type=float, required=True,
                        help="optional mask-shape flow loss weight")
    parser.add_argument("--lambda_leak", type=float, required=True)
    parser.add_argument("--lambda_attn", type=float, required=True)
    parser.add_argument("--lambda_feat", type=float, required=True)

    parser.add_argument("--embed_dim", type=int, required=True,
                        help="total feature dimension, equal to 3*C")
    parser.add_argument("--window_size", type=int, required=True,
                        help="window size used by window attention")
    parser.add_argument("--tcb_depths", type=parse_int_tuple, required=True,
                        help="comma-separated TCB block depths")
    parser.add_argument("--sab_depths", type=parse_int_tuple, required=True,
                        help="comma-separated SAB block depths")
    parser.add_argument("--tcb_num_heads", type=parse_int_tuple, required=True,
                        help="comma-separated TCB attention heads")
    parser.add_argument("--sab_num_heads", type=parse_int_tuple, required=True,
                        help="comma-separated SAB attention heads")
    parser.add_argument("--prm_stage_blocks", type=parse_int_tuple, required=True,
                        help="comma-separated PRM stage block counts")
    parser.add_argument("--prm_stage_heads", type=parse_int_tuple, required=True,
                        help="comma-separated PRM stage attention heads")
    parser.add_argument("--mlp_ratio", type=float, required=True)
    parser.add_argument("--drop_path_rate", type=float, required=True)
    parser.add_argument("--fusion_drop_path_rate", type=float, required=True)
    parser.add_argument("--no_cuda", action="store_true")

    return parser.parse_args()


def cache_name(cache_choice):
    if cache_choice == 0:
        return "none"
    if cache_choice == 1:
        return "bin"
    if cache_choice == 2:
        return "in_memory"
    raise ValueError(f"Unsupported cache_choice: {cache_choice}")


def build_model(args):
    upscale = 4
    window_size = args.window_size
    height = (128 // upscale // window_size + 1) * window_size
    width = (128 // upscale // window_size + 1) * window_size
    return ReAlignHDR(
        img_size=(height, width),
        embed_dim=args.embed_dim,
        tcb_depths=args.tcb_depths,
        sab_depths=args.sab_depths,
        tcb_num_heads=args.tcb_num_heads,
        sab_num_heads=args.sab_num_heads,
        prm_stage_blocks=args.prm_stage_blocks,
        prm_stage_heads=args.prm_stage_heads,
        window_size=args.window_size,
        mlp_ratio=args.mlp_ratio,
        drop_path_rate=args.drop_path_rate,
        fusion_drop_path_rate=args.fusion_drop_path_rate,
    )


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def remap_state_dict_keys(state_dict):
    remapped = {}
    replacements = (
        ("module.", ""),
        ("texture_branch.", "tcb."),
        ("artifact_branch.", "sab."),
        ("fusion.", "prm."),
    )
    for key, value in state_dict.items():
        new_key = key
        for old, new in replacements:
            if old == "module.":
                if new_key.startswith(old):
                    new_key = new_key[len(old):]
            else:
                new_key = new_key.replace(old, new)
        remapped[new_key] = value
    return remapped


def load_model_checkpoint(model, ckpt_path, strict=True):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = remap_state_dict_keys(extract_state_dict(checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return checkpoint, missing, unexpected


def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def zero_masks_like(mask1, mask2, mask3):
    return torch.zeros_like(mask1), torch.zeros_like(mask2), torch.zeros_like(mask3)


def attention_distillation_losses(teacher_attention, student_attention, eps=1e-8):
    leak_loss = None
    attn_loss = None
    count = 0

    for path in ("src1", "src3"):
        teacher_records = teacher_attention[path]
        student_records = student_attention[path]
        if len(teacher_records) != len(student_records):
            raise ValueError(
                f"Attention record mismatch for {path}: "
                f"teacher={len(teacher_records)}, student={len(student_records)}"
            )

        for teacher_record, student_record in zip(teacher_records, student_records):
            gate = teacher_record["gate"].to(dtype=student_record["attention"].dtype)
            teacher_attn = teacher_record["attention"].detach()
            student_attn = student_record["attention"]

            invalid = 1.0 - gate
            leak = (invalid * student_attn).abs().mean()

            teacher_valid = teacher_attn * gate
            student_valid = student_attn * gate
            teacher_sum = teacher_valid.sum(dim=-1, keepdim=True)
            student_sum = student_valid.sum(dim=-1, keepdim=True)
            valid_rows = teacher_sum > eps

            teacher_prob = teacher_valid / (teacher_sum + eps)
            student_prob = student_valid / (student_sum + eps)
            kl = teacher_prob * (
                torch.log(teacher_prob + eps) - torch.log(student_prob + eps)
            )
            kl = kl.sum(dim=-1, keepdim=True)
            kl = torch.where(valid_rows, kl, torch.zeros_like(kl)).sum() / (
                valid_rows.sum().clamp_min(1)
            )

            leak_loss = leak if leak_loss is None else leak_loss + leak
            attn_loss = kl if attn_loss is None else attn_loss + kl
            count += 1

    if count == 0:
        raise ValueError("No attention records were returned for distillation")
    return leak_loss / count, attn_loss / count


def feature_distillation_loss(teacher_features, student_features):
    loss = 0.0
    for key in ("SAB", "TCB", "fuse"):
        loss = loss + F.l1_loss(student_features[key], teacher_features[key].detach())
    return loss / 3.0


def train_one_epoch(args, teacher, student, device, train_loader, optimizer, epoch, criterion, logger, writer):
    teacher.eval()
    student.train()

    for batch_idx, batch_data in enumerate(tqdm(train_loader, ncols=80)):
        batch_ldrs = [ldr.to(device, non_blocking=True) for ldr in batch_data["inputs"]]
        batch_ldrs = torch.cat(batch_ldrs, dim=1)
        I1, I2, I3 = torch.chunk(batch_ldrs, 3, dim=1)
        label = batch_data["label"].to(device, non_blocking=True)

        masks = batch_data["masks"].to(device, non_blocking=True)
        masks_bin = (masks > 0).float()
        mask1, mask2, mask3 = masks_bin[:, 0], masks_bin[:, 1], masks_bin[:, 2]
        zero1, zero2, zero3 = zero_masks_like(mask1, mask2, mask3)

        with torch.no_grad():
            _, _, _, teacher_aux = teacher(
                batch_ldrs, mask1, mask2, mask3, return_features=True
            )

        pred, flow_1_2, flow_3_2, student_aux = student(
            batch_ldrs, zero1, zero2, zero3, return_features=True
        )

        rec_loss, rec_dict = criterion(
            pred, label, I1, I2, I3, zero1, zero2, zero3, flow_1_2, flow_3_2
        )
        leak_loss, attn_loss = attention_distillation_losses(
            teacher_aux["attention"], student_aux["attention"]
        )
        feat_loss = feature_distillation_loss(
            teacher_aux["features"], student_aux["features"]
        )
        loss = (
            rec_loss
            + args.lambda_leak * leak_loss
            + args.lambda_attn * attn_loss
            + args.lambda_feat * feat_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            msg = (
                f"Train Student Epoch: {epoch} [{batch_idx * args.batch_size}/"
                f"{len(train_loader.dataset)}]\t"
                f"Loss: {loss.item():.6f}\t"
                f"Rec: {rec_loss.item():.6f}\t"
                f"Leak: {leak_loss.item():.6f}\t"
                f"Attn: {attn_loss.item():.6f}\t"
                f"Feat: {feat_loss.item():.6f}"
            )
            logger.info(msg)
            writer.add_scalar("train/loss", loss.item(), batch_idx + (epoch - 1) * len(train_loader))
            writer.add_scalar("train/loss_rec", rec_loss.item(), batch_idx + (epoch - 1) * len(train_loader))
            writer.add_scalar("train/loss_leak", leak_loss.item(), batch_idx + (epoch - 1) * len(train_loader))
            writer.add_scalar("train/loss_attn", attn_loss.item(), batch_idx + (epoch - 1) * len(train_loader))
            writer.add_scalar("train/loss_feat", feat_loss.item(), batch_idx + (epoch - 1) * len(train_loader))
            for key, value in rec_dict.items():
                if hasattr(value, "item"):
                    writer.add_scalar(f"train/{key}", value.item(), batch_idx + (epoch - 1) * len(train_loader))


def evaluate(args, model, device, test_loader):
    model.eval()
    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()
    pu_psnr = AverageMeter()
    pu_ssim = AverageMeter()
    pu_psnr_list = []
    pu_ssim_list = []

    with torch.no_grad():
        for batch_data in tqdm(test_loader, ncols=80):
            batch_ldrs = [ldr.to(device, non_blocking=True) for ldr in batch_data["inputs"]]
            batch_ldrs = torch.cat(batch_ldrs, dim=1)
            label = batch_data["label"].to(device, non_blocking=True)
            masks = batch_data["masks"].to(device, non_blocking=True)
            masks_bin = (masks > 0).float()

            padded_image, padded_masks = pad_image(batch_ldrs, masks_bin, args.patch_size, args.patch_size)
            zero_masks = torch.zeros_like(padded_masks)
            pred_img, _, _ = model(padded_image, zero_masks[:, 0], zero_masks[:, 1], zero_masks[:, 2])

            _, _, orig_h, orig_w = label.size()
            pred_img = pred_img[:, :, :orig_h, :orig_w]

            mse_l = F.mse_loss(label, pred_img)
            scene_psnr_l = 20 * torch.log10(1.0 / torch.sqrt(mse_l))
            scene_ssim_l = pytorch_ssim.ssim(label, pred_img)

            label_mu = range_compressor(label)
            pred_img_mu = range_compressor(pred_img)
            mse_mu = F.mse_loss(label_mu, pred_img_mu)
            scene_psnr_mu = 20 * torch.log10(1.0 / torch.sqrt(mse_mu))
            scene_ssim_mu = pytorch_ssim.ssim(label_mu, pred_img_mu)

            pred_np = pred_img.detach().cpu().numpy()
            label_np = label.detach().cpu().numpy()
            batch_pu = [
                pu21_metrics(
                    pred_np[i],
                    label_np[i],
                    peak_luminance=args.pu_peak_luminance,
                )
                for i in range(pred_np.shape[0])
            ]
            scene_pu_psnr = float(np.mean([item[0] for item in batch_pu]))
            scene_pu_ssim = float(np.mean([item[1] for item in batch_pu]))

            psnr_l.update(scene_psnr_l)
            ssim_l.update(scene_ssim_l)
            psnr_mu.update(scene_psnr_mu)
            ssim_mu.update(scene_ssim_mu)
            pu_psnr.update(scene_pu_psnr)
            pu_ssim.update(scene_pu_ssim)
            pu_psnr_list.append(scene_pu_psnr)
            pu_ssim_list.append(scene_pu_ssim)

    pu_psnr_mean, pu_psnr_std = mean_std(pu_psnr_list)
    pu_ssim_mean, pu_ssim_std = mean_std(pu_ssim_list)

    return {
        "psnr_l": psnr_l.avg,
        "ssim_l": ssim_l.avg,
        "psnr_mu": psnr_mu.avg,
        "ssim_mu": ssim_mu.avg,
        "pu21_psnr": pu_psnr.avg,
        "pu21_ssim": pu_ssim.avg,
        "pu21_psnr_mean": pu_psnr_mean,
        "pu21_psnr_std": pu_psnr_std,
        "pu21_ssim_mean": pu_ssim_mean,
        "pu21_ssim_std": pu_ssim_std,
    }


def save_checkpoint(path, epoch, model, optimizer, lr_scheduler, best_metric):
    payload = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_metric": best_metric,
    }
    if lr_scheduler is not None:
        payload["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(payload, path)


def main():
    args = get_args()
    if args.seed is not None:
        set_random_seed(args.seed)

    os.makedirs(args.logdir, exist_ok=True)
    ckpt_dir = os.path.join(args.logdir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.logdir, "tensorboard", "curve"))
    logger = get_logger("train_student", args.logdir)
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    from dataset.training_dataset import Training_Dataset, Testing_Dataset
    from loss.loss import loss_total

    cache = cache_name(args.cache_choice)
    train_dataset = Training_Dataset(
        root_dir=args.dataset_dir,
        patch_size=args.train_patch_size,
        repeat=args.repeat,
        cache=cache,
        train_path=args.train_path,
        exposure_file_name=args.exposure_file_name,
        ldr_folder_name=args.ldr_folder_name,
        label_file_name=args.label_file_name,
        mask_npy_name=args.mask_npy_name,
        ldr_prefix=args.ldr_prefix or "",
        foreground_patch_prob=args.foreground_patch_prob,
        foreground_min_pixels=args.foreground_min_pixels,
        foreground_sample_attempts=args.foreground_sample_attempts,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_dataset = Testing_Dataset(
        root_dir=args.dataset_dir,
        patch_size=args.patch_size,
        repeat=1,
        cache=cache,
        train_path=args.test_path,
        exposure_file_name=args.exposure_file_name,
        ldr_folder_name=args.ldr_folder_name,
        label_file_name=args.label_file_name,
        mask_npy_name=args.mask_npy_name,
        ldr_prefix=args.ldr_prefix or "",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        pin_memory=True,
    )
    logger.info(f"Training dataset size: {len(train_dataset)}, testing dataset size: {len(test_dataset)}")

    teacher = build_model(args).to(device)
    student = build_model(args).to(device)

    _, missing, unexpected = load_model_checkpoint(teacher, args.teacher_ckpt, strict=True)
    logger.info(f"Loaded teacher: missing={missing}, unexpected={unexpected}")
    freeze_model(teacher)

    if args.student_resume:
        checkpoint, missing, unexpected = load_model_checkpoint(student, args.student_resume, strict=True)
        logger.info(f"Loaded student resume: missing={missing}, unexpected={unexpected}")
        args.start_epoch = int(checkpoint.get("epoch", args.start_epoch - 1)) + 1
    elif args.init_student_from_teacher:
        _, missing, unexpected = load_model_checkpoint(student, args.teacher_ckpt, strict=True)
        logger.info(f"Initialized student from teacher: missing={missing}, unexpected={unexpected}")

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    lr_scheduler = None
    if args.lr_decay:
        lr_scheduler = MyLR(
            optimizer,
            T_max=args.epochs,
            phase1_epoch=args.phase1_epochs,
            eta_min=args.eta_min,
        )

    if args.student_resume:
        checkpoint = torch.load(args.student_resume, map_location="cpu")
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if args.lr_decay and lr_scheduler is not None and "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    criterion = loss_total(lambda_fft=args.lambda_fft,
                           lambda_perc=args.lambda_perc,
                           lambda_census=args.lambda_census,
                           lambda_shape=args.lambda_shape).to(device)
    best_pu21_psnr = float("-inf")

    for epoch in range(args.start_epoch, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"===> Student Epoch: {epoch}/{args.epochs}, lr={current_lr:.7f}")
        writer.add_scalar("train/lr", current_lr, epoch)

        train_one_epoch(args, teacher, student, device, train_loader, optimizer, epoch, criterion, logger, writer)

        if args.lr_decay:
            lr_scheduler.step()

        if epoch % args.eval_interval == 0:
            metrics = evaluate(args, student, device, test_loader)
            logger.info(
                "Mask-free student eval: "
                f"PSNR_l={metrics['psnr_l']:.4f}, PSNR_mu={metrics['psnr_mu']:.4f}, "
                f"SSIM_l={metrics['ssim_l']:.4f}, SSIM_mu={metrics['ssim_mu']:.4f}, "
                f"PU21-PSNR={metrics['pu21_psnr']:.4f}, PU21-SSIM={metrics['pu21_ssim']:.4f}"
            )
            logger.info(
                "PU21-absolute student eval: "
                f"PU-PSNR={metrics['pu21_psnr_mean']:.4f} ± {metrics['pu21_psnr_std']:.4f}, "
                f"PU-SSIM={metrics['pu21_ssim_mean']:.4f} ± {metrics['pu21_ssim_std']:.4f}, "
                f"peak_luminance={args.pu_peak_luminance:.4f}"
            )
            for key, value in metrics.items():
                writer.add_scalar(f"test/{key}", value, epoch)

            if metrics["pu21_psnr"] > best_pu21_psnr:
                best_pu21_psnr = metrics["pu21_psnr"]
                save_checkpoint(
                    os.path.join(ckpt_dir, "best_student.pth"),
                    epoch,
                    student,
                    optimizer,
                    lr_scheduler,
                    best_pu21_psnr,
                )

        if epoch % args.save_interval == 0:
            save_checkpoint(
                os.path.join(ckpt_dir, f"epoch_{epoch}.pth"),
                epoch,
                student,
                optimizer,
                lr_scheduler,
                best_pu21_psnr,
            )

    writer.close()


if __name__ == "__main__":
    main()
