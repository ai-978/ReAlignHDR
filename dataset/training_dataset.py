#-*- coding:utf-8 -*-
import os
import os.path as osp
import sys
sys.path.append('..')
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.utils import *
import pickle
import random
import numpy as np

def _load_masks_common(mask_source, ldr_num):
    if isinstance(mask_source, (list, tuple)):
        masks = [np.load(p, allow_pickle=True) for p in mask_source]
    else:
        masks = np.load(mask_source, allow_pickle=True)
        if masks.ndim == 4 and masks.shape[0] == ldr_num:
            masks = [m for m in masks]
        else:
            masks = [masks]

    norm_masks = []
    for m in masks:
        m = (m > 0).astype(np.float32)
        
        if m.ndim == 2:
            m = m[..., None]
        elif m.ndim == 3 and m.shape[-1] != 1 and m.shape[0] != ldr_num:
            # (H, W, N) -> split last dim
            m = np.moveaxis(m, -1, 0)
            norm_masks.extend([mm[..., None] if mm.ndim == 2 else mm for mm in m])
            continue
        norm_masks.append(m)

    if len(norm_masks) == 1 and ldr_num > 1:
        norm_masks = norm_masks * ldr_num

    norm_masks = [mm if mm.ndim == 3 else mm[None, ...] for mm in norm_masks]
    stacked = np.stack([mm if mm.shape[-1] == 1 else mm[..., None] for mm in norm_masks], axis=0)
    if stacked.shape[0] != ldr_num and stacked.shape[1] == ldr_num:
        stacked = stacked[:, 0]
    if stacked.shape[0] != ldr_num:
        stacked = stacked[:ldr_num]
    return stacked


class Training_Dataset(Dataset):

    def __init__(self, root_dir, patch_size, repeat, cache,
                 train_path,
                 exposure_file_name,
                 ldr_folder_name, 
                 label_file_name,
                 foreground_patch_prob,
                 foreground_min_pixels,
                 foreground_sample_attempts,
                 mask_npy_name=None,
                 ldr_prefix = ""):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.repeat = repeat
        self.cache = cache
        self.label_file_name = label_file_name
        if not 0 <= foreground_patch_prob <= 1:
            raise ValueError("foreground_patch_prob must be in [0, 1]")
        if foreground_min_pixels < 1:
            raise ValueError("foreground_min_pixels must be >= 1")
        if foreground_sample_attempts < 1:
            raise ValueError("foreground_sample_attempts must be >= 1")
        self.foreground_patch_prob = foreground_patch_prob
        self.foreground_min_pixels = foreground_min_pixels
        self.foreground_sample_attempts = foreground_sample_attempts

        self.scenes_dir = osp.join(root_dir, train_path)
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            scene_dir = os.path.join(self.scenes_dir, self.scenes_list[scene])
            exposure_file_path = os.path.join(scene_dir, exposure_file_name)
            if ldr_folder_name is None:
                ldr_file_path = list_all_files_sorted_with_prefix(scene_dir, '.tif', ldr_prefix)
                mask_file_path = list_all_files_sorted_with_prefix(scene_dir, '.npy', ldr_prefix)
            else:
                ldr_file_path = list_all_files_sorted_with_prefix(os.path.join(scene_dir, ldr_folder_name), '.tif', ldr_prefix)
                mask_file_path = list_all_files_sorted_with_prefix(os.path.join(scene_dir, ldr_folder_name), '.npy', ldr_prefix)
            label_path = scene_dir

            single_mask_path = os.path.join(scene_dir, mask_npy_name) if mask_npy_name else None
            mask_path = single_mask_path if single_mask_path and os.path.exists(single_mask_path) else mask_file_path
            
            if cache == 'none':
                self.image_list += [[exposure_file_path, ldr_file_path, label_path, mask_path]]

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_dir),
                    '_bin_' + os.path.basename(root_dir))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                exposure_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_exposure.pkl')
                if not os.path.exists(exposure_bin_file):
                    with open(exposure_bin_file, 'wb') as f:
                        pickle.dump(read_expo_times(exposure_file_path), f)
                    print('dump', exposure_bin_file)
                ldrs_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_ldr.pkl')
                if not os.path.exists(ldrs_bin_file):
                    with open(ldrs_bin_file, 'wb') as f:
                        pickle.dump(read_images(ldr_file_path), f)
                    print('dump', ldrs_bin_file)
                label_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_label.pkl')
                if not os.path.exists(label_bin_file):
                    with open(label_bin_file, 'wb') as f:
                        pickle.dump(read_label(label_path, label_file_name), f)
                    print('dump', label_bin_file)
                masks_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_mask.pkl')
                if not os.path.exists(masks_bin_file):
                    with open(masks_bin_file, 'wb') as f:
                        pickle.dump(self._load_masks(mask_path, len(ldr_file_path)), f)
                    print('dump', masks_bin_file)
                self.image_list.append([exposure_bin_file, ldrs_bin_file, label_bin_file, masks_bin_file])

            elif cache == 'in_memory':
                # Read exposure times
                expoTimes = read_expo_times(exposure_file_path)
                # Read LDR images
                ldr_images = read_images(ldr_file_path)
                # Read HDR label
                label = read_label(label_path, label_file_name)
                # Read mask
                masks = self._load_masks(mask_path, len(ldr_file_path))
                self.image_list.append([expoTimes, ldr_images, label, masks])

    def __getitem__(self, index):

        # calculate index
        index = index % len(self.scenes_list)

        if self.cache == 'none':
            # Read exposure times
            expoTimes = read_expo_times(self.image_list[index][0])

            # Read LDR images
            ldr_images = read_images(self.image_list[index][1])
            
            # Read HDR label
            label = read_label(self.image_list[index][2], self.label_file_name)

            # Read mask
            masks = self._load_masks(self.image_list[index][3], len(ldr_images))
        
        elif self.cache == 'bin':
            with open(self.image_list[index][0], 'rb') as f:
                expoTimes = pickle.load(f)
            with open(self.image_list[index][1], 'rb') as f:
                ldr_images = pickle.load(f)
            with open(self.image_list[index][2], 'rb') as f:
                label = pickle.load(f)
            with open(self.image_list[index][3], 'rb') as f:
                masks = pickle.load(f)

        elif self.cache == 'in_memory':
            expoTimes, ldr_images, label, masks = self.image_list[index]
        
        # Random crop
        ldr_images, label, masks = self._random_crop(ldr_images, label, masks, self.patch_size)

        # data augmentation
        ldr_images, label, masks = self._augment(ldr_images, label, masks)
        
        # ldr images process
        pre_imgs = [ldr_to_hdr(ldr_images[i], expoTimes[i], 2.2) for i in range(len(ldr_images))]    
        pre_imgs = [np.concatenate((pre_imgs[i], ldr_images[i]), 2) for i in range(len(ldr_images))]
        imgs = [pre_imgs[i].astype(np.float32).transpose(2, 0, 1) for i in range(len(ldr_images))]
        imgs = [torch.from_numpy(imgs[i]) for i in range(len(ldr_images))]        
        
        # hdr image process
        label = label.astype(np.float32).transpose(2, 0, 1)
        label = torch.from_numpy(label)

        # mask process
        masks = masks.astype(np.float32).transpose(0, 3, 1, 2)  # (N,1,H,W)
        masks = torch.from_numpy(masks)
        
        sample = {
            'inputs': imgs, 
            'label': label,
            'masks': masks,
            }
        return sample

    def __len__(self):
        return len(self.scenes_list)*self.repeat

    def _load_masks(self, mask_source, ldr_num):
        return _load_masks_common(mask_source, ldr_num)

    def _random_crop_origin(self, H, W, patch_size):
        h = random.randint(0, max(0, H - patch_size))
        w = random.randint(0, max(0, W - patch_size))
        return h, w

    def _crop_has_foreground(self, masks, h, w, patch_size):
        patch = masks[:, h:h+patch_size, w:w+patch_size, :]
        return np.count_nonzero(patch > 0) >= self.foreground_min_pixels

    def _sample_foreground_crop_origin(self, masks, H, W, patch_size):
        foreground_map = np.any(masks > 0, axis=(0, 3))
        foreground_pixels = np.argwhere(foreground_map)
        if foreground_pixels.size == 0:
            return None

        max_h = max(0, H - patch_size)
        max_w = max(0, W - patch_size)
        for _ in range(self.foreground_sample_attempts):
            y, x = foreground_pixels[random.randrange(len(foreground_pixels))]
            h_min = max(0, int(y) - patch_size + 1)
            h_max = min(int(y), max_h)
            w_min = max(0, int(x) - patch_size + 1)
            w_max = min(int(x), max_w)
            h = random.randint(h_min, h_max) if h_min <= h_max else 0
            w = random.randint(w_min, w_max) if w_min <= w_max else 0
            if self._crop_has_foreground(masks, h, w, patch_size):
                return h, w
        return None

    def _sample_background_crop_origin(self, masks, H, W, patch_size):
        for _ in range(self.foreground_sample_attempts):
            h, w = self._random_crop_origin(H, W, patch_size)
            if not self._crop_has_foreground(masks, h, w, patch_size):
                return h, w
        return None

    def _sample_balanced_crop_origin(self, masks, H, W, patch_size):
        if random.random() < self.foreground_patch_prob:
            origin = self._sample_foreground_crop_origin(masks, H, W, patch_size)
        else:
            origin = self._sample_background_crop_origin(masks, H, W, patch_size)
        return origin if origin is not None else self._random_crop_origin(H, W, patch_size)

    def _random_crop(self, ldr_images, label, masks, patch_size):
        _, H, W, _ = ldr_images.shape 
        h, w = self._sample_balanced_crop_origin(masks, H, W, patch_size)
        ldr_images = ldr_images[:, h:h+patch_size, w:w+patch_size, :]
        label = label[h:h+patch_size, w:w+patch_size, :]
        masks = masks[:, h:h+patch_size, w:w+patch_size, :]
        return ldr_images, label, masks

    def _augment(self, ldr_images, label, masks):
        if random.random() > 0.5:
            if random.random() > 0.5:
                ldr_images = ldr_images[:, ::-1, :, :]
                label = label[::-1, :, :]
                masks = masks[:, ::-1, :, :]
            else:
                ldr_images = ldr_images[:, :, ::-1, :]
                label = label[:, ::-1, :]
                masks = masks[:, :, ::-1, :]
        return ldr_images, label, masks
    

class Validing_Dataset(Dataset):
    def __init__(self, root_dir, patch_size, repeat, cache,
                 train_path,
                 exposure_file_name,
                 ldr_folder_name, 
                 label_file_name,
                 mask_npy_name=None,
                 ldr_prefix = ""):
        self.root_dir = root_dir  # /Kalantari
        self.patch_size = patch_size  # 128
        self.repeat = repeat
        self.cache = cache
        self.label_file_name = label_file_name

        self.scenes_dir = osp.join(root_dir, train_path)  # /Kalantari/Training
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            scene_dir = os.path.join(self.scenes_dir, self.scenes_list[scene])
            exposure_file_path = os.path.join(scene_dir, exposure_file_name)
            if ldr_folder_name is None:
                ldr_file_path = list_all_files_sorted_with_prefix(scene_dir, '.tif', ldr_prefix)
                mask_file_path = list_all_files_sorted_with_prefix(scene_dir, '.npy', ldr_prefix)
            else:
                ldr_file_path = list_all_files_sorted_with_prefix(os.path.join(scene_dir, ldr_folder_name), '.tif', ldr_prefix)
                mask_file_path = list_all_files_sorted_with_prefix(os.path.join(scene_dir, ldr_folder_name), '.npy', ldr_prefix)
            label_path = scene_dir

            single_mask_path = os.path.join(scene_dir, mask_npy_name) if mask_npy_name else None
            mask_path = single_mask_path if single_mask_path and os.path.exists(single_mask_path) else mask_file_path
            
            if cache == 'none':
                self.image_list += [[exposure_file_path, ldr_file_path, label_path, mask_path]]

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_dir),
                    '_bin_valid_' + os.path.basename(root_dir))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                exposure_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_exposure.pkl')
                if not os.path.exists(exposure_bin_file):
                    with open(exposure_bin_file, 'wb') as f:
                        pickle.dump(read_expo_times(exposure_file_path), f)
                    print('dump', exposure_bin_file)
                ldrs_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_ldr.pkl')
                if not os.path.exists(ldrs_bin_file):
                    with open(ldrs_bin_file, 'wb') as f:
                        pickle.dump(read_images(ldr_file_path), f)
                    print('dump', ldrs_bin_file)
                label_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_label.pkl')
                if not os.path.exists(label_bin_file):
                    with open(label_bin_file, 'wb') as f:
                        pickle.dump(read_label(label_path, label_file_name), f)
                    print('dump', label_bin_file)
                masks_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_mask.pkl')
                if not os.path.exists(masks_bin_file):
                    with open(masks_bin_file, 'wb') as f:
                        pickle.dump(_load_masks_common(mask_path, len(ldr_file_path)), f)
                    print('dump', masks_bin_file)
                self.image_list.append([exposure_bin_file, ldrs_bin_file, label_bin_file, masks_bin_file])

            elif cache == 'in_memory':
                # Read exposure times
                expoTimes = read_expo_times(exposure_file_path)
                # Read LDR images
                ldr_images = read_images(ldr_file_path)
                # Read HDR label
                label = read_label(label_path, label_file_name)
                # Read mask
                masks = _load_masks_common(mask_path, len(ldr_file_path))
                self.image_list.append([expoTimes, ldr_images, label, masks])

    def __getitem__(self, index):

        # calculate index
        index = index % len(self.scenes_list)

        if self.cache == 'none':
            # Read exposure times
            expoTimes = read_expo_times(self.image_list[index][0])

            # Read LDR images
            ldr_images = read_images(self.image_list[index][1])
            
            # Read HDR label
            label = read_label(self.image_list[index][2], self.label_file_name)

            # Read mask
            masks = _load_masks_common(self.image_list[index][3], len(ldr_images))
        
        elif self.cache == 'bin':
            with open(self.image_list[index][0], 'rb') as f:
                expoTimes = pickle.load(f)
            with open(self.image_list[index][1], 'rb') as f:
                ldr_images = pickle.load(f)
            with open(self.image_list[index][2], 'rb') as f:
                label = pickle.load(f)
            with open(self.image_list[index][3], 'rb') as f:
                masks = pickle.load(f)

        elif self.cache == 'in_memory':
            expoTimes, ldr_images, label, masks = self.image_list[index]
        
        # ldr images process
        pre_imgs = [ldr_to_hdr(ldr_images[i], expoTimes[i], 2.2) for i in range(len(ldr_images))]    
        pre_imgs = [np.concatenate((pre_imgs[i], ldr_images[i]), 2) for i in range(len(ldr_images))]
        imgs = [pre_imgs[i].astype(np.float32).transpose(2, 0, 1) for i in range(len(ldr_images))]
        imgs = [torch.from_numpy(imgs[i]) for i in range(len(ldr_images))]        
        
        # hdr image process
        label = label.astype(np.float32).transpose(2, 0, 1)
        label = torch.from_numpy(label)

        masks = masks.astype(np.float32).transpose(0, 3, 1, 2)
        masks = torch.from_numpy(masks)
        
        sample = {
            'inputs': imgs, 
            'label': label,
            'masks': masks,
            }
        return sample

    def __len__(self):
        return len(self.scenes_list)*self.repeat


class Testing_Dataset(Dataset):
    def __init__(self, root_dir, patch_size, repeat, cache,
                 train_path,
                 exposure_file_name,
                 ldr_folder_name, 
                 label_file_name,
                 mask_npy_name=None,
                 ldr_prefix = ""):
        self.root_dir = root_dir  # /Kalantari
        self.patch_size = patch_size  # 128
        self.repeat = repeat
        self.cache = cache
        self.label_file_name = label_file_name

        self.scenes_dir = osp.join(root_dir, train_path)  # /Kalantari/Test
        self.scenes_list = sorted(os.listdir(self.scenes_dir))

        # self.scenes_list = self.scenes_list[::5]  # <<<<<<<<<<<================= 

        self.image_list = []
        for scene in range(len(self.scenes_list)):
            scene_dir = os.path.join(self.scenes_dir, self.scenes_list[scene])
            exposure_file_path = os.path.join(scene_dir, exposure_file_name)
            if ldr_folder_name is None:
                ldr_file_path = list_all_files_sorted_with_prefix(scene_dir, '.tif', ldr_prefix)
                mask_file_path = list_all_files_sorted_with_prefix(scene_dir, '.npy', ldr_prefix)
            else:
                ldr_file_path = list_all_files_sorted_with_prefix(os.path.join(scene_dir, ldr_folder_name), '.tif',ldr_prefix)
                mask_file_path = list_all_files_sorted_with_prefix(os.path.join(scene_dir, ldr_folder_name), '.npy', ldr_prefix)
            label_path = scene_dir
            single_mask_path = os.path.join(scene_dir, mask_npy_name) if mask_npy_name else None
            mask_path = single_mask_path if single_mask_path and os.path.exists(single_mask_path) else mask_file_path
            
            if cache == 'none':
                self.image_list += [[exposure_file_path, ldr_file_path, label_path, mask_path]]

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_dir),
                    '_bin_test_' + os.path.basename(root_dir))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                exposure_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_exposure.pkl')
                if not os.path.exists(exposure_bin_file):
                    with open(exposure_bin_file, 'wb') as f:
                        pickle.dump(read_expo_times(exposure_file_path), f)
                    print('dump', exposure_bin_file)
                ldrs_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_ldr.pkl')
                if not os.path.exists(ldrs_bin_file):
                    with open(ldrs_bin_file, 'wb') as f:
                        pickle.dump(read_images(ldr_file_path), f)
                    print('dump', ldrs_bin_file)
                label_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_label.pkl')
                if not os.path.exists(label_bin_file):
                    with open(label_bin_file, 'wb') as f:
                        pickle.dump(read_label(label_path, label_file_name), f)
                    print('dump', label_bin_file)
                masks_bin_file = os.path.join(bin_root, self.scenes_list[scene] + '_mask.pkl')
                if not os.path.exists(masks_bin_file):
                    with open(masks_bin_file, 'wb') as f:
                        pickle.dump(_load_masks_common(mask_path, len(ldr_file_path)), f)
                    print('dump', masks_bin_file)
                self.image_list.append([exposure_bin_file, ldrs_bin_file, label_bin_file, masks_bin_file])

            elif cache == 'in_memory':
                # Read exposure times
                expoTimes = read_expo_times(exposure_file_path)
                # Read LDR images
                ldr_images = read_images(ldr_file_path)
                # Read HDR label
                label = read_label(label_path, label_file_name)
                # Read mask
                masks = _load_masks_common(mask_path, len(ldr_file_path))
                self.image_list.append([expoTimes, ldr_images, label, masks])

    def __getitem__(self, index):

        # calculate index
        index = index % len(self.scenes_list)

        scence = self.scenes_list[index]
        name = os.path.basename(scence)

        if self.cache == 'none':
            # Read exposure times
            expoTimes = read_expo_times(self.image_list[index][0])

            # Read LDR images
            ldr_images = read_images(self.image_list[index][1])
            
            # Read HDR label
            label = read_label(self.image_list[index][2], self.label_file_name)

            # Read mask
            masks = _load_masks_common(self.image_list[index][3], len(ldr_images))
        
        elif self.cache == 'bin':
            with open(self.image_list[index][0], 'rb') as f:
                expoTimes = pickle.load(f)
            with open(self.image_list[index][1], 'rb') as f:
                ldr_images = pickle.load(f)
            with open(self.image_list[index][2], 'rb') as f:
                label = pickle.load(f)
            with open(self.image_list[index][3], 'rb') as f:
                masks = pickle.load(f)

        elif self.cache == 'in_memory':
            expoTimes, ldr_images, label, masks = self.image_list[index]
        
        # ldr images process
        pre_imgs = [ldr_to_hdr(ldr_images[i], expoTimes[i], 2.2) for i in range(len(ldr_images))]    
        pre_imgs = [np.concatenate((pre_imgs[i], ldr_images[i]), 2) for i in range(len(ldr_images))]
        imgs = [pre_imgs[i].astype(np.float32).transpose(2, 0, 1) for i in range(len(ldr_images))]
        imgs = [torch.from_numpy(imgs[i]) for i in range(len(ldr_images))]        
        
        # hdr image process
        label = label.astype(np.float32).transpose(2, 0, 1)
        label = torch.from_numpy(label)

        masks = masks.astype(np.float32).transpose(0, 3, 1, 2)
        masks = torch.from_numpy(masks)

        sample = {
            'inputs': imgs, 
            'label': label,
            'masks': masks,
            'name': name,
            }
        return sample

    def __len__(self):
        return len(self.scenes_list)*self.repeat
