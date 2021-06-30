"""
Modified from https://github.com/seoungwugoh/STM/blob/master/dataset.py
"""

import os
from os import path

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import json

from dataset.range_transform import im_normalization
from dataset.util import all_to_onehot


class YouTubeVOSTestDataset(Dataset):
    def __init__(self, data_root, split, mask_root=None, metadata=None, in_range=None):
        self.image_dir = path.join(data_root, 'all_frames', split+'_all_frames', 'JPEGImages')
        if mask_root is None:
            self.mask_dir = path.join(data_root, split, 'Annotations')
        else:
            self.mask_dir = mask_root
        with open(metadata) as f:
            self.metadata = json.load(f)
        if in_range is not None:
            self.metadata = self.metadata[in_range]
        self.videos = []
        self.shape = {}
        self.frames = {}
        self.meta_propagate = []
        vid_list = sorted(self.metadata.keys())
        for vid in self.metadata.keys():
            for eid in self.metadata[vid].keys():
                self.meta_propagate.append({'video_id': vid, 'exp_id': eid, 'mask_ids': self.metadata[vid][eid]})
        # Pre-reading
        for vid in vid_list:
            frames = sorted(os.listdir(os.path.join(self.image_dir, vid)))
            self.frames[vid] = frames

            self.videos.append(vid)
            first_frame = np.array(Image.open(path.join(self.image_dir, vid, frames[0])))
            self.shape[vid] = np.shape(first_frame)[:2]
            # first_mask = os.listdir(path.join(self.mask_dir, vid))[0]
            # _mask = np.array(Image.open(path.join(self.mask_dir, vid, first_mask)).convert("P"))
            # self.shape[vid] = np.shape(_mask)

        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
            transforms.Resize(480, interpolation=Image.BICUBIC),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(480, interpolation=Image.NEAREST),
        ])

    def __getitem__(self, idx):
        data_info = self.meta_propagate[idx]
        video = data_info['video_id']
        eid = data_info['exp_id']
        mask_ids = data_info['mask_ids']
        info = {}
        info['name'] = video
        info['exp_id'] = eid
        info['num_objects'] = 0
        info['frames'] = self.frames[video] 
        info['size'] = self.shape[video] # Real sizes
        info['gt_obj'] = {} # Frames with labelled objects

        vid_im_path = path.join(self.image_dir, video)
        vid_gt_path = path.join(self.mask_dir, video, eid)

        skip = False
        frames = self.frames[video]

        images = []
        masks = []
        for i, f in enumerate(frames):
            img = Image.open(path.join(vid_im_path, f)).convert('RGB')
            images.append(self.im_transform(img))
            fid = f.split('.')[0]
            mask = np.zeros(self.shape[video])
            if fid in mask_ids:
                mask_file = path.join(vid_gt_path, f'{fid}.png')
                mask = np.array(Image.open(mask_file).resize(self.shape[video][::-1], resample=Image.NEAREST).convert('P'), dtype=np.uint8)
            masks.append(mask)
            if (np.sum(mask) != 0):
                this_labels = np.unique(masks[-1])
                this_labels = this_labels[this_labels!=0]
                info['gt_obj'][i] = this_labels
        
        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)
        
        # Construct the forward and backward mapping table for labels
        labels = np.unique(masks).astype(np.uint8)
        labels = labels[labels!=0]
        info['label_convert'] = {}
        info['label_backward'] = {}
        idx = 1
        for l in labels:
            info['label_convert'][l] = idx
            info['label_backward'][idx] = l
            idx += 1
        masks = torch.from_numpy(all_to_onehot(masks, labels)).float()

        # Resize to 480p
        masks = self.mask_transform(masks)
        masks = masks.unsqueeze(2)

        info['labels'] = labels

        data = {
            'rgb': images,
            'gt': masks,
            'info': info,
            'skip': skip
        }

        return data

    def __len__(self):
        return len(self.meta_propagate)
