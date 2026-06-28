# -*- coding:utf-8 -*-
import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.training_dataset import Training_Dataset, Validing_Dataset, Testing_Dataset
from dataset.test_dataset import Test_Dataset
from loss.loss import loss_total
from models.ReAlignHDR import ReAlignHDR
from utils.utils import * 
from utils.pu21 import pu21_metrics
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from lr_scheduler.mylr import MyLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pytorch_ssim

def parse_int_tuple(value):
    if isinstance(value, tuple):
        return value
    return tuple(int(item.strip()) for item in value.split(',') if item.strip())

def pad_image(image, masks, patch_size, stride_size):
    _, _, h, w = image.size()
    pad_h = (stride_size - (h - patch_size) % stride_size) % stride_size
    pad_w = (stride_size - (w - patch_size) % stride_size) % stride_size
    padding = (0, pad_w, 0, pad_h)
    padded_image = F.pad(image, padding, mode='reflect')
    # masks: [B, 3, 1, H, W] -> pad spatial dims only
    masks_bhw = masks.view(masks.shape[0]*masks.shape[1], 1, masks.shape[-2], masks.shape[-1])
    masks_bhw = F.pad(masks_bhw, padding, mode='replicate')
    padded_masks = masks_bhw.view(masks.shape[0], masks.shape[1], 1, masks_bhw.shape[-2], masks_bhw.shape[-1])
    return padded_image, padded_masks


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return float(values.mean()), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def get_args():
    parser = argparse.ArgumentParser(description='All Settings')
    parser.add_argument('--logdir', type=str, required=True,
                        help='target log directory')
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help='dataset directory')
    parser.add_argument('--train_path', type=str, required=True,
                        help='train split path')
    parser.add_argument('--test_path', type=str, required=True,
                        help='test split path')
    parser.add_argument('--exposure_file_name', type=str, required=True,
                        help='exposure file name')
    parser.add_argument('--ldr_prefix', type=str,
                        help='ldr tif prefix string')
    parser.add_argument('--ldr_folder_name', type=str,
                        help='ldr folder name')
    parser.add_argument('--label_file_name', type=str, required=True,
                        help='label file name')
    parser.add_argument('--mask_npy_name', type=str,
                        help='mask npy name; omit to use per-LDR .npy files')
    
    # Training and Test Settings
    parser.add_argument('--train_patch_size', type=int, required=True,
                        help='patch size for training')
    parser.add_argument('--patch_size', type=int, required=True,
                        help='patch size for test')
    parser.add_argument('--repeat', type=int, required=True,
                        help='number of repeat for training dataset')
    parser.add_argument('--num_workers', type=int, required=True, metavar='N',
                        help='number of workers to fetch data for training')
    parser.add_argument('--test_num_workers', type=int, required=True, metavar='N',
                        help='number of workers to fetch data for test')
    parser.add_argument('--start_epoch', type=int, required=True, metavar='N',
                        help='start epoch of training')
    parser.add_argument('--epochs', type=int, required=True, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--phase1_epochs', type=int, required=True, metavar='N',
                        help='number of scheduler phase epochs')
    parser.add_argument('--batch_size', type=int, required=True, metavar='N',
                        help='training batch size')
    parser.add_argument('--test_batch_size', type=int, required=True, metavar='N',
                        help='testing batch size')
    parser.add_argument('--log_interval', type=int, required=True, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str,
                        help='load model from a .pth file')
    parser.add_argument('--seed', type=int, required=True, metavar='S',
                        help='random seed')
    parser.add_argument('--cache_choice', type=int, required=True,
                        help='cache for dataloader(0: none, 1: bin, 2: in_memory)')
    parser.add_argument('--foreground_patch_prob', type=float, required=True,
                        help='target probability of sampling a patch containing foreground mask pixels')
    parser.add_argument('--foreground_min_pixels', type=int, required=True,
                        help='minimum foreground pixels required for a foreground patch')
    parser.add_argument('--foreground_sample_attempts', type=int, required=True,
                        help='maximum crop attempts for foreground/background balanced sampling')
    parser.add_argument('--momentum', type=float, required=True, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--lr', type=float, required=True, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr_decay', action='store_true',
                        help='learning rate decay or not')
    parser.add_argument('--eta_min', type=float, required=True,
                        help='minimum learning rate for the scheduler')
    parser.add_argument('--lambda_fft', type=float, required=True,
                        help='weight of the FFT loss')
    parser.add_argument('--lambda_perc', type=float, required=True,
                        help='weight of the perceptual loss')
    parser.add_argument('--lambda_census', type=float, required=True,
                        help='weight of the census flow loss')
    parser.add_argument('--lambda_shape', type=float, required=True,
                        help='optional mask-shape flow loss weight')
    parser.add_argument('--pu_peak_luminance', type=float, required=True,
                        help='display peak luminance in nits for PU21-absolute metrics')
    parser.add_argument('--embed_dim', type=int, required=True,
                        help='total feature dimension, equal to 3*C')
    parser.add_argument('--window_size', type=int, required=True,
                        help='window size used by window attention')
    parser.add_argument('--tcb_depths', type=parse_int_tuple, required=True,
                        help='comma-separated TCB block depths, e.g. 5 or 4,5,5,4')
    parser.add_argument('--sab_depths', type=parse_int_tuple, required=True,
                        help='comma-separated SAB block depths, e.g. 5 or 4,5,5,4')
    parser.add_argument('--tcb_num_heads', type=parse_int_tuple, required=True,
                        help='comma-separated TCB attention heads')
    parser.add_argument('--sab_num_heads', type=parse_int_tuple, required=True,
                        help='comma-separated SAB attention heads')
    parser.add_argument('--prm_stage_blocks', type=parse_int_tuple, required=True,
                        help='comma-separated PRM stage block counts')
    parser.add_argument('--prm_stage_heads', type=parse_int_tuple, required=True,
                        help='comma-separated PRM stage attention heads')
    parser.add_argument('--mlp_ratio', type=float, required=True)
    parser.add_argument('--drop_path_rate', type=float, required=True)
    parser.add_argument('--fusion_drop_path_rate', type=float, required=True)

    # Other Settings
    parser.add_argument('--no_cuda', action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--init_weights', action='store_true',
                        help='init model weights')
    parser.add_argument('--init_gain', type=float, required=True,
                        help='gain used when --init_weights is enabled')
    parser.add_argument('--is_freeze', action='store_true',
                        help='freeze partial parameters or not')
    return parser.parse_args()

def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    for batch_idx, batch_data in enumerate(tqdm(train_loader, ncols=80)):
        batch_ldrs = [ldr.to(device) for ldr in batch_data['inputs']]
        batch_ldrs = torch.cat(batch_ldrs, dim=1)  # [B, 18, H, W]

        I1, I2, I3 = torch.chunk(batch_ldrs, 3, dim=1)  # [B,6,H,W]
        label = batch_data['label'].to(device)  # [B,3,H,W]
        masks = batch_data['masks'].to(device)  # [B,3,1,H,W], 0/1/2/...
        masks_bin = (masks > 0).float()
        mask1_bin, mask2_bin, mask3_bin = masks_bin[:, 0], masks_bin[:, 1], masks_bin[:, 2]

        pred, flow_1_2, flow_3_2 = model(batch_ldrs, mask1_bin, mask2_bin, mask3_bin)
        loss, loss_dict = criterion(pred, label, I1, I2, I3, mask1_bin, mask2_bin, mask3_bin, flow_1_2, flow_3_2)

        # backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            loss_recon = loss_dict.get('loss_recon', torch.tensor(0.0))
            loss_le = loss_dict.get('loss_le', torch.tensor(0.0))
            loss_perc = loss_dict.get('loss_perc', torch.tensor(0.0))
            loss_shape = loss_dict.get('loss_shape', torch.tensor(0.0))
            loss_census = loss_dict.get('loss_census', torch.tensor(0.0))
            
            logger_train.info('Train Epoch: {} [{}/{} ({:.0f} %)]\tLoss: {:.6f}\t'
                  'L1: {:.6f}\tFFT: {:.6f}\tPerc: {:.6f}\tShape: {:.6f}\tCensus: {:.6f}'.format(
                epoch,
                batch_idx * args.batch_size,
                len(train_loader.dataset),
                100. * batch_idx * args.batch_size / len(train_loader.dataset),
                loss.item(),
                loss_recon.item() if hasattr(loss_recon, 'item') else loss_recon,
                loss_le.item() if hasattr(loss_le, 'item') else loss_le,
                loss_perc.item() if hasattr(loss_perc, 'item') else loss_perc,
                loss_shape.item() if hasattr(loss_shape, 'item') else loss_shape,
                loss_census.item() if hasattr(loss_census, 'item') else loss_census,
            ))
            
            tb_writer.add_scalar('train/loss', loss.item(), batch_idx+(epoch-1)*len(train_loader.dataset))
            tb_writer.add_scalar('train/loss_recon', loss_recon.item() if hasattr(loss_recon, 'item') else loss_recon, batch_idx+(epoch-1)*len(train_loader.dataset))
            tb_writer.add_scalar('train/loss_le', loss_le.item() if hasattr(loss_le, 'item') else loss_le, batch_idx+(epoch-1)*len(train_loader.dataset))

def test_single_img(args, model, img_dataset, device):
    dataloader = DataLoader(dataset=img_dataset, batch_size=args.test_batch_size, num_workers=args.test_num_workers, shuffle=False)
    with torch.no_grad():
        for batch_data in dataloader:
            # dataloader
            batch_ldrs = [ldr.to(device) for ldr in batch_data['inputs']]
            batch_ldrs = torch.cat(batch_ldrs, dim=1)
            masks = batch_data['masks'].to(device)
            masks_bin = (masks > 0).float()
            pred_hdr, _, _ = model(batch_ldrs, masks_bin[:, 0], masks_bin[:, 1], masks_bin[:, 2])
            img_dataset.update_result(torch.squeeze(pred_hdr.detach().cpu()).numpy().astype(np.float32))
    pred, label = img_dataset.rebuild_result()
    return pred, label

def test(args, model, device, optimizer, lr_scheduler, epoch, test_loader, ckpt_dir):
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
        for batch_idx, batch_data in enumerate(tqdm(test_loader, ncols=80)):
            # dataloader
            batch_ldrs = [ldr.to(device) for ldr in batch_data['inputs']]
            batch_ldrs = torch.cat(batch_ldrs, dim=1)
            label = batch_data['label'].to(device)
            masks = batch_data['masks'].to(device)
            masks_bin = (masks > 0).float()

            padded_image, padded_masks = pad_image(batch_ldrs, masks_bin, args.patch_size, args.patch_size)
            pred_img, _, _ = model(padded_image, padded_masks[:, 0], padded_masks[:, 1], padded_masks[:, 2])

            _, _, orig_h, orig_w = label.size()
            pred_img = pred_img[:, :, :orig_h, :orig_w]

            pred_np = pred_img.detach().cpu().numpy().astype(np.float32)
            label_np = label.detach().cpu().numpy().astype(np.float32)
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

            mse_l =  F.mse_loss(label,pred_img)
            scene_psnr_l = (20 * torch.log10(1.0 / torch.sqrt(mse_l)))
            scene_ssim_l = pytorch_ssim.ssim(label, pred_img)

            label_mu = range_compressor(label)
            pred_img_mu = range_compressor(pred_img)
            mse_mu =  F.mse_loss(label_mu, pred_img_mu)
            scene_psnr_mu = (20 * torch.log10(1.0 / torch.sqrt(mse_mu)))
            scene_ssim_mu = pytorch_ssim.ssim(label_mu, pred_img_mu)

            psnr_l.update(scene_psnr_l)
            ssim_l.update(scene_ssim_l)
            psnr_mu.update(scene_psnr_mu)
            ssim_mu.update(scene_ssim_mu) 
            pu_psnr.update(scene_pu_psnr)
            pu_ssim.update(scene_pu_ssim)
            pu_psnr_list.append(scene_pu_psnr)
            pu_ssim_list.append(scene_pu_ssim)

    if best_metric['psnr_l']['value'] < psnr_l.avg:
        best_metric['psnr_l']['value'] = psnr_l.avg
        best_metric['psnr_l']['epoch'] = epoch
    if best_metric['psnr_mu']['value'] < psnr_mu.avg:
        best_metric['psnr_mu']['value'] = psnr_mu.avg
        best_metric['psnr_mu']['epoch'] = epoch
    if best_metric['ssim_l']['value'] < ssim_l.avg:
        best_metric['ssim_l']['value'] = ssim_l.avg
        best_metric['ssim_l']['epoch'] = epoch
    if best_metric['ssim_mu']['value'] < ssim_mu.avg:
        best_metric['ssim_mu']['value'] = ssim_mu.avg
        best_metric['ssim_mu']['epoch'] = epoch
    if best_metric['pu_psnr']['value'] < pu_psnr.avg:
        best_metric['pu_psnr']['value'] = pu_psnr.avg
        best_metric['pu_psnr']['epoch'] = epoch
    if best_metric['pu_ssim']['value'] < pu_ssim.avg:
        best_metric['pu_ssim']['value'] = pu_ssim.avg
        best_metric['pu_ssim']['epoch'] = epoch

    pu_psnr_mean, pu_psnr_std = mean_std(pu_psnr_list)
    pu_ssim_mean, pu_ssim_std = mean_std(pu_ssim_list)

    logger_train.info('Epoch:' + str(epoch))
    logger_train.info('Test set: Average PSNR: {:.4f}, PSNR_mu: {:.4f}, SSIM_l: {:.4f}, SSIM_mu: {:.4f}, PU21-PSNR: {:.4f}, PU21-SSIM: {:.4f} (peak luminance: {:.1f} nits)\n'.format(
        psnr_l.avg,
        psnr_mu.avg,
        ssim_l.avg,
        ssim_mu.avg,
        pu_psnr.avg,
        pu_ssim.avg,
        args.pu_peak_luminance
        ))
    logger_train.info('PU21-absolute PU-PSNR: {:.4f} ± {:.4f}  PU-SSIM: {:.4f} ± {:.4f} (peak luminance: {:.1f} nits)'.format(
        pu_psnr_mean,
        pu_psnr_std,
        pu_ssim_mean,
        pu_ssim_std,
        args.pu_peak_luminance
    ))
    logger_valid.info('==Best==\tPSNR_l: {:.4f}/epoch: {}\t PSNR_mu: {:.4f}/epoch: {} \t SSIM_l: {:.4f}/epoch: {}\t SSIM_mu: {:.4f}/epoch: {}\t PU21-PSNR: {:.4f}/epoch: {}\t PU21-SSIM: {:.4f}/epoch: {}'.format(
        best_metric['psnr_l']['value'], best_metric['psnr_l']['epoch'],
        best_metric['psnr_mu']['value'], best_metric['psnr_mu']['epoch'],
        best_metric['ssim_l']['value'], best_metric['ssim_l']['epoch'],
        best_metric['ssim_mu']['value'], best_metric['ssim_mu']['epoch'],
        best_metric['pu_psnr']['value'], best_metric['pu_psnr']['epoch'],
        best_metric['pu_ssim']['value'], best_metric['pu_ssim']['epoch']
    ))

    save_dict = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict()
    }
    torch.save(save_dict, os.path.join(ckpt_dir, 'epoch_{:d}.pth'.format(epoch)))
    tb_writer.add_scalar('test/psnr_l', psnr_l.avg, epoch)
    tb_writer.add_scalar('test/psnr_mu', psnr_mu.avg, epoch)
    tb_writer.add_scalar('test/ssim_l', ssim_l.avg, epoch)
    tb_writer.add_scalar('test/ssim_mu', ssim_mu.avg, epoch)
    tb_writer.add_scalar('test/pu21_psnr', pu_psnr.avg, epoch)
    tb_writer.add_scalar('test/pu21_ssim', pu_ssim.avg, epoch)

def main():
    print('===> Init settings')
    args = get_args()
    
    if args.seed is not None:
        set_random_seed(args.seed)
    
    logdir = args.logdir
    tensorboard_dir_curve = os.path.join(logdir, 'tensorboard','curve')
    tensorboard_dir_figure = os.path.join(logdir, 'tensorboard','figure')
    ckpt_dir = os.path.join(logdir, 'ckpt')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(tensorboard_dir_curve):
        os.makedirs(tensorboard_dir_curve)
    if not os.path.exists(tensorboard_dir_figure):
        os.makedirs(tensorboard_dir_figure)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    global logger_train
    logger_train = get_logger('train', logdir)
    global logger_valid
    logger_valid = get_logger('valid', logdir)
    global tb_writer
    tb_writer = SummaryWriter(os.path.join(tensorboard_dir_curve))
    global tb_figure
    tb_figure = SummaryWriter(os.path.join(tensorboard_dir_figure))

    args_dict = vars(args)
    for key, value in args_dict.items():
        logger_train.info(f'{key}: {value}')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    print('===> Loading datasets')
    if args.cache_choice == 0:
        cache = 'none'
        print('===> No cache')
    elif args.cache_choice == 1:
        cache = 'bin'
        print('===> Cache bin')
    elif args.cache_choice == 2:
        cache = 'in_memory'
        print('===> Cache in_memory')

    train_dataset = Training_Dataset(root_dir=args.dataset_dir, 
                                     patch_size=args.train_patch_size, 
                                     repeat=args.repeat, cache=cache, 
                                     train_path=args.train_path, 
                                     exposure_file_name=args.exposure_file_name, 
                                     ldr_folder_name=args.ldr_folder_name, 
                                     label_file_name=args.label_file_name,
                                     mask_npy_name=args.mask_npy_name,
                                     ldr_prefix=args.ldr_prefix or "",
                                     foreground_patch_prob=args.foreground_patch_prob,
                                     foreground_min_pixels=args.foreground_min_pixels,
                                     foreground_sample_attempts=args.foreground_sample_attempts)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, 
                              pin_memory=True)  
    trainset_size = len(train_loader.dataset)

    test_dataset = Testing_Dataset(root_dir=args.dataset_dir, 
                                    patch_size=args.patch_size, 
                                    repeat=1, cache=cache, 
                                    train_path=args.test_path, 
                                    exposure_file_name=args.exposure_file_name, 
                                    ldr_folder_name=args.ldr_folder_name, 
                                    label_file_name=args.label_file_name,
                                    mask_npy_name=args.mask_npy_name,
                                    ldr_prefix=args.ldr_prefix or "")
    
    test_loader = DataLoader(test_dataset, batch_size=1, 
                              shuffle=False, num_workers=1, 
                              pin_memory=True) 
    
    testset_size = len(test_loader.dataset)

    print('===> Training dataset size: {},Testing dataset size: {}.'.format(trainset_size, testset_size))

    upscale = 4
    window_size = args.window_size
    height = (128 // upscale // window_size + 1) * window_size
    width = (128 // upscale // window_size + 1) * window_size
    model = ReAlignHDR(img_size=(height, width),
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
                       fusion_drop_path_rate=args.fusion_drop_path_rate)
    
    # init
    if args.init_weights:
        init_weights(model, init_type='normal', gain=args.init_gain)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)# 1e-8
    
    # lr_scheduler
    lr_scheduler = None
    if args.lr_decay:
        lr_scheduler = MyLR(optimizer, T_max=args.epochs, phase1_epoch=args.phase1_epochs, eta_min=args.eta_min)

    model.to(device)
    model = nn.DataParallel(model)

    # load checkpoint
    if args.resume and os.path.isfile(args.resume):
        if args.is_freeze:
            print("===> Loading checkpoint from: {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = 1
            model.load_state_dict(checkpoint['state_dict'])
            model = freeze_model(model=model, not_freeze_list=['module.conv_first.0.weight', 'module.conv_first.0.bias'])
            # optimizer
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), \
                                         lr=args.lr, betas=(0.9, 0.999), eps=1e-8)# 1e-8
            lr_scheduler = None
            if args.lr_decay:
                lr_scheduler = MyLR(optimizer, T_max=args.epochs, phase1_epoch = args.phase1_epochs, 
                                    eta_min=args.eta_min)
            print("===> Start fine-tuning.")
        else:
            print("===> Loading checkpoint from: {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.lr_decay and lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("===> Loaded checkpoint: epoch {}".format(checkpoint['epoch']))
    else:
        print("===> No checkpoint is founded.")

    # model complexity
    from ptflops import get_model_complexity_info
    profile_model = model.module if isinstance(model, nn.DataParallel) else model

    class _ProfileWrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net.eval()

        def forward(self, x):
            b, _, h, w = x.shape
            dummy = torch.zeros((b, 1, h, w), device=x.device)
            return self.net(x, dummy, dummy, dummy)

    with torch.no_grad():
        flops, params = get_model_complexity_info(
            _ProfileWrapper(profile_model), (18, 128, 128),
            as_strings=True, print_per_layer_stat=False, verbose=False
        )
        logger_train.info(f'### flops: {flops}, params: {params}.')
        print('## Flops: ', flops, ', Params: ', params)

    # loss  
    criterion = loss_total(lambda_fft=args.lambda_fft,
                           lambda_perc=args.lambda_perc,
                           lambda_census=args.lambda_census,
                           lambda_shape=args.lambda_shape).to(device)
    # metrics
    global best_metric
    best_metric = {'psnr_l': {'value': 0., 'epoch': 0},
                   'psnr_mu': {'value': 0., 'epoch': 0},
                   'ssim_l': {'value': 0., 'epoch': 0},
                   'ssim_mu': {'value': 0., 'epoch': 0},
                   'pu_psnr': {'value': 0., 'epoch': 0},
                   'pu_ssim': {'value': 0., 'epoch': 0}}

    for epoch in range(args.start_epoch, args.epochs + 1):
        logger_train.info(f'===> Epoch: {epoch}/{args.epochs}')
        print(f'===> Epoch: {epoch}/{args.epochs}')
        
        for param_group in optimizer.param_groups:
            logger_train.info("Learning rate is: [{:1.7f}] ==".format(param_group['lr']))
            tb_writer.add_scalar('train/lr', param_group['lr'], epoch)
            print("Learning rate is: [{:1.7f}] ==".format(param_group['lr']))
        
        train(args, model, device, train_loader, optimizer, epoch, criterion)
        if args.lr_decay:
            lr_scheduler.step()
        
        print(f"==> start test of epoch {epoch}.")
        test(args, model, device, optimizer, lr_scheduler, epoch, test_loader, ckpt_dir)


if __name__ == '__main__':
    main()
