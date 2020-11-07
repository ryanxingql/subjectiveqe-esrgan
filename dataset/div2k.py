import glob
import random
import torch
from pathlib import Path
import numpy as np
from cv2 import cv2
from torch.utils import data as data
from utils import FileClient, paired_random_crop, augment, totensor, import_yuv


def _read_img_bytes(img_bytes):
    """
    Read image bytes encoded in LMDB.

    update: 20-10-28
    """
    img_np = np.frombuffer(img_bytes, np.uint8)
    img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)  # (H W [BGR])
    img_np = (img_np / 255.).astype(np.float32)
    return img_np


class DIV2KTrainingSet(data.Dataset):
    """
    Train with LMDB.
    
    update: 20-10-28
    """
    def __init__(self, opts_dict):
        super().__init__()

        self.opts_dict = opts_dict
        
        # dataset paths
        dataroot = Path('data/DIV2K/')
        self.gt_root = dataroot / self.opts_dict['gt_path']
        self.lq_root = dataroot / self.opts_dict['lq_path']
        self.meta_info_path = self.gt_root / 'meta_info.txt'

        with open(self.meta_info_path, 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root, 
            self.gt_root
            ]
        self.io_opts_dict['client_keys'] = ['lq', 'gt']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )

        key = self.keys[index]
        img_bytes = self.file_client.get(key, 'gt')
        img_gt = _read_img_bytes(img_bytes)  # (H W [BGR])
        img_bytes = self.file_client.get(key, 'lq')
        img_lq = _read_img_bytes(img_bytes)  # (H W [BGR])

        # randomly crop
        gt_size = self.opts_dict['gt_size']
        img_gt, img_lq = paired_random_crop(
            img_gt, img_lq, gt_size,
            )

        # flip, rotate
        img_batch = [img_lq, img_gt] # gt joint augmentation with lq
        img_batch = augment(
            img_batch, self.opts_dict['use_flip'], self.opts_dict['use_rot']
            )

        # to tensor
        img_batch = totensor(img_batch)  # ([RGB] H W)
        img_lq, img_gt = img_batch[:]

        return {
            'lq': img_lq,
            'gt': img_gt,
            }

    def __len__(self):
        return len(self.keys)


def _read_img(img_path):
    img_np = cv2.imread(str(img_path))  # (H W [BGR])
    img_np = (img_np / 255.).astype(np.float32)
    return img_np


class DIV2KTestSet(data.Dataset):
    """
    Test with disk IO.
    
    update: 20-10-28
    """
    def __init__(self, opts_dict):
        super().__init__()

        self.opts_dict = opts_dict

        # dataset paths
        dataroot = Path('data/DIV2K/')
        self.gt_root = dataroot / self.opts_dict['gt_path']
        self.lq_root = dataroot / self.opts_dict['lq_path']

        # record data info for loading
        self.data_info = {
            'gt_path': [],
            'lq_path': [],
            'index_vid': [],
            'name_vid': [],
            }

        gt_path_list = list(self.gt_root.glob('*.png'))
        self.vid_num = len(gt_path_list)
        
        for idx_vid, gt_vid_path in enumerate(gt_path_list):
            name_vid = gt_vid_path.stem
            lq_vid_path = self.lq_root / (name_vid + '.png')
            self.data_info['index_vid'].append(idx_vid)
            self.data_info['gt_path'].append(gt_vid_path)
            self.data_info['lq_path'].append(lq_vid_path)
            self.data_info['name_vid'].append(name_vid)

    def __getitem__(self, index):
        img_gt = _read_img(self.data_info['gt_path'][index])  # (H W [BGR])
        img_lq = _read_img(self.data_info['lq_path'][index])  # (H W [BGR])

        # no any augmentation

        # to tensor
        img_batch = [img_lq, img_gt]
        img_batch = totensor(img_batch)  # ([RGB] H W)
        img_lq, img_gt = img_batch[:]

        return {
            'lq': img_lq,
            'gt': img_gt,
            'name_vid': self.data_info['name_vid'][index], 
            'index_vid': self.data_info['index_vid'][index], 
            }

    def __len__(self):
        return self.vid_num
    