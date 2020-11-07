import glob
import random
import torch
from pathlib import PurePath
import numpy as np
from cv2 import cv2
from torch.utils import data as data
from utils import FileClient, paired_random_crop, augment, totensor, import_yuv


def _bytes2img(img_bytes):
    img_np = np.frombuffer(img_bytes, np.uint8)
    img_np = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)  # (H W)
    img_np = np.expand_dims(img_np, 2)  # (H W Y)
    img_np = img_np.astype(np.float32) / 255.
    return img_np


class Vimeo90K4thTrainingSet(data.Dataset):
    """Train with LMDB."""

    def __init__(self, opts_dict):
        super().__init__()

        self.opts_dict = opts_dict
        
        # dataset paths
        vimeo_dataroot = PurePath('data/Vimeo-90K/')
        self.gt_root = vimeo_dataroot / self.opts_dict['gt_path']
        self.lq_root = vimeo_dataroot / self.opts_dict['lq_path']
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
        img_gt = _bytes2img(img_bytes)  # (H W Y)
        img_bytes = self.file_client.get(key, 'lq')
        img_lq = _bytes2img(img_bytes)  # (H W Y)

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
        img_batch = totensor(img_batch)
        img_lq, img_gt = img_batch[:]

        return {
            'lq': img_lq,  # (Y H W)
            'gt': img_gt,  # (Y H W)
            }

    def __len__(self):
        return len(self.keys)


class Vimeo90K4thTestSet(data.Dataset):
    """Test with disk IO."""
    
    def __init__(self, opts_dict):
        super().__init__()

        self.opts_dict = opts_dict

        # dataset paths
        vimeo_dataroot = PurePath('data/Vimeo-90K/')
        self.gt_root = vimeo_dataroot / self.opts_dict['gt_path']
        self.lq_root = vimeo_dataroot / self.opts_dict['lq_path']
        self.meta_info_path = vimeo_dataroot / self.opts_dict['meta_path']

        # record data info for loading
        self.data_info = {
            'gt_path': [],
            'lq_path': [],
            'gt_index': [],
            'lq_index': [],
            'h': [],
            'w': [],
            'index_vid': [],
            'name_vid': [],
            }

        gt_path_list = []
        meta_fp = open(self.meta_info_path, 'r')
        while True:
            new_line = meta_fp.readline().split('\n')[0]
            if new_line == '':
                break
            vid_name = new_line.split('/')[0] + '_' + new_line.split('/')[1]
            gt_path = self.gt_root / (vid_name + '.yuv')
            gt_path_list.append(gt_path)
        
        self.vid_num = len(gt_path_list)
        for idx_vid, gt_vid_path in enumerate(gt_path_list):
            name_vid = gt_vid_path.stem
            w, h = 448, 256
            lq_vid_path = self.lq_root / (name_vid + '.yuv')
            self.data_info['index_vid'].append(idx_vid)
            self.data_info['gt_path'].append(gt_vid_path)
            self.data_info['lq_path'].append(lq_vid_path)
            self.data_info['name_vid'].append(name_vid)
            self.data_info['w'].append(w)
            self.data_info['h'].append(h)
            self.data_info['gt_index'].append(3)
            self.data_info['lq_index'].append(3)

    def __getitem__(self, index):
        # get gt frame
        img = import_yuv(
            seq_path=self.data_info['gt_path'][index], 
            yuv_type='444p', 
            h=self.data_info['h'][index],
            w=self.data_info['w'][index],
            tot_frm=1,
            start_frm=self.data_info['gt_index'][index],
            only_y=True
            )
        img_gt = np.expand_dims(
            np.squeeze(img), 2
            ).astype(np.float32) / 255.  # (H W Y)

        # get lq frames
        img = import_yuv(
            seq_path=self.data_info['lq_path'][index], 
            yuv_type='444p', 
            h=self.data_info['h'][index],
            w=self.data_info['w'][index],
            tot_frm=1,
            start_frm=self.data_info['lq_index'][index],
            only_y=True
            )
        img_lq = np.expand_dims(
            np.squeeze(img), 2
            ).astype(np.float32) / 255.  # (H W Y)

        # no any augmentation

        # to tensor
        img_batch = [img_lq, img_gt]
        img_batch = totensor(img_batch)
        img_lq, img_gt = img_batch[:]

        return {
            'lq': img_lq,  # (Y H W)
            'gt': img_gt,  # (Y H W)
            'name_vid': self.data_info['name_vid'][index], 
            'index_vid': self.data_info['index_vid'][index], 
            }

    def __len__(self):
        return len(self.data_info['gt_path'])

    def get_vid_num(self):
        return self.vid_num


def _yuvbytes2img(img_bytes):
    img_np = np.frombuffer(img_bytes, np.uint8)
    img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)  # (H W [VUY])
    img_np = img_np.astype(np.float32) / 255.
    return img_np


class Vimeo90K4thYUVTrainingSet(data.Dataset):
    """
    Train with LMDB.
    
    update: 20-10-24
    """
    def __init__(self, opts_dict):
        super().__init__()

        self.opts_dict = opts_dict
        
        # dataset paths
        vimeo_dataroot = PurePath('data/Vimeo-90K/')
        self.gt_root = vimeo_dataroot / self.opts_dict['gt_path']
        self.lq_root = vimeo_dataroot / self.opts_dict['lq_path']
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
        img_gt = _yuvbytes2img(img_bytes)  # (H W [YUV])
        img_bytes = self.file_client.get(key, 'lq')
        img_lq = _yuvbytes2img(img_bytes)  # (H W [YUV])

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
        img_batch = totensor(img_batch)
        img_lq, img_gt = img_batch[:]

        return {
            'lq': img_lq,  # ([YUV] H W)
            'gt': img_gt,  # ([YUV] H W)
            }

    def __len__(self):
        return len(self.keys)


class Vimeo90K4thYUVTestSet(data.Dataset):
    """
    Test with disk IO.
    
    update: 20-10-24
    """
    
    def __init__(self, opts_dict):
        super().__init__()

        self.opts_dict = opts_dict

        # dataset paths
        vimeo_dataroot = PurePath('data/Vimeo-90K/')
        self.gt_root = vimeo_dataroot / self.opts_dict['gt_path']
        self.lq_root = vimeo_dataroot / self.opts_dict['lq_path']
        self.meta_info_path = vimeo_dataroot / self.opts_dict['meta_path']

        # record data info for loading
        self.data_info = {
            'gt_path': [],
            'lq_path': [],
            'gt_index': [],
            'lq_index': [],
            'h': [],
            'w': [],
            'index_vid': [],
            'name_vid': [],
            }

        gt_path_list = []
        meta_fp = open(self.meta_info_path, 'r')
        while True:
            new_line = meta_fp.readline().split('\n')[0]
            if new_line == '':
                break
            vid_name = new_line.split('/')[0] + '_' + new_line.split('/')[1]
            gt_path = self.gt_root / (vid_name + '.yuv')
            gt_path_list.append(gt_path)
        
        self.vid_num = len(gt_path_list)
        for idx_vid, gt_vid_path in enumerate(gt_path_list):
            name_vid = gt_vid_path.stem
            w, h = 448, 256
            lq_vid_path = self.lq_root / (name_vid + '.yuv')
            self.data_info['index_vid'].append(idx_vid)
            self.data_info['gt_path'].append(gt_vid_path)
            self.data_info['lq_path'].append(lq_vid_path)
            self.data_info['name_vid'].append(name_vid)
            self.data_info['w'].append(w)
            self.data_info['h'].append(h)
            self.data_info['gt_index'].append(3)
            self.data_info['lq_index'].append(3)

    def __getitem__(self, index):
        # get gt frame
        img = import_yuv(
            seq_path=self.data_info['gt_path'][index], 
            yuv_type='444p', 
            h=self.data_info['h'][index],
            w=self.data_info['w'][index],
            tot_frm=1,
            start_frm=self.data_info['gt_index'][index],
            only_y=False
            )
        img_gt = np.concatenate(img, axis=0)
        img_gt = np.transpose(img_gt, (1, 2, 0))
        img_gt = img_gt.astype(np.float32) / 255.  # (H W [YUV])

        # get lq frames
        img = import_yuv(
            seq_path=self.data_info['lq_path'][index], 
            yuv_type='444p', 
            h=self.data_info['h'][index],
            w=self.data_info['w'][index],
            tot_frm=1,
            start_frm=self.data_info['lq_index'][index],
            only_y=False
            )
        img_lq = np.concatenate(img, axis=0)
        img_lq = np.transpose(img_lq, (1, 2, 0))
        img_lq = img_lq.astype(np.float32) / 255.  # (H W [YUV])

        # no any augmentation

        # to tensor
        img_batch = [img_lq, img_gt]
        img_batch = totensor(img_batch)
        img_lq, img_gt = img_batch[:]

        return {
            'lq': img_lq,  # ([YUV] H W)
            'gt': img_gt,  # ([YUV] H W)
            'name_vid': self.data_info['name_vid'][index], 
            'index_vid': self.data_info['index_vid'][index], 
            }

    def __len__(self):
        return len(self.data_info['gt_path'])

    def get_vid_num(self):
        return self.vid_num