#-*- coding:utf-8 -*-
import os
import os.path as osp
import sys
sys.path.append('..')
from utils.utils import *
from dataset.image_dataset import Image_Dataset


def Test_Dataset(root_dir, patch_size, mode,
                 test_path,
                 exposure_file_name,
                 ldr_folder_name, 
                 label_file_name,
                 mask_npy_name=None):
    scenes_dir = osp.join(root_dir, test_path)
    scenes_list = sorted(os.listdir(scenes_dir))
    ldr_list = []
    label_list = []
    expo_times_list = []
    mask_npy_list = []
    if mode == 'train':
        stride_size = patch_size
    elif mode == 'test':
        stride_size = None
    for scene in range(len(scenes_list)):
        exposure_file_path = os.path.join(scenes_dir, scenes_list[scene], exposure_file_name)
        if ldr_folder_name is None:
            ldr_file_path = list_all_files_sorted_with_prefix(os.path.join(scenes_dir, scenes_list[scene]), '.tif')
            mask_file_path = list_all_files_sorted_with_prefix(os.path.join(scenes_dir, scenes_list[scene]), '.npy')
        else:
            ldr_file_path = list_all_files_sorted_with_prefix(os.path.join(scenes_dir, scenes_list[scene], ldr_folder_name), '.tif')
            mask_file_path = list_all_files_sorted_with_prefix(os.path.join(scenes_dir, scenes_list[scene], ldr_folder_name), '.npy')
        label_path = os.path.join(scenes_dir, scenes_list[scene])
        single_mask_path = os.path.join(scenes_dir, scenes_list[scene], mask_npy_name) if mask_npy_name else None
        mask_npy_path = single_mask_path if single_mask_path and os.path.exists(single_mask_path) else mask_file_path
        expo_times_list += [exposure_file_path]
        mask_npy_list += [mask_npy_path]
        ldr_list += [ldr_file_path]
        label_list += [label_path]

    for ldr_dir, label_dir, expo_times_dir, mask_npy_dir in zip(ldr_list, label_list, expo_times_list, mask_npy_list):
        yield Image_Dataset(ldr_dir, label_dir, expo_times_dir, mask_npy_dir, patch_size, label_file_name, stride_size)
