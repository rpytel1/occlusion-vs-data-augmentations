# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import logging
import random

import torch
import numpy as np
import cv2
from lib.dataset.coco import COCODataset
from lib.dataset.transformations.multi_keypoint_transformation import MultiKeypointTransformation
from lib.dataset.transformations.random_image_transform import RandomImageTransform
from lib.dataset.transformations.targeted_blurring import TargetedBlurring

from lib.dataset.transformations.targeted_cutmix import TargetedCutmix
from lib.dataset.transformations.targeted_cutout import Cutout

from lib.utils.transforms import affine_transform, get_affine_transform, fliplr_joints
from train import get_image_dir_path

logger = logging.getLogger(__name__)


class ContrastiveCOCODataset(COCODataset):
    '''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
	"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''

    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.occlusion_transforms_keypoint = []
        self.occlusion_transforms_parts = []
        ## REMEMBER TO NOT USE JITTERING WHEN USING CONTRASTIVE!
        if cfg.CONTRASTIVE.CUTMIX:
            self.occlusion_transforms_parts.append(
                RandomImageTransform(
                    TargetedCutmix(self.get_image_dir_path(cfg), self._get_ann_file_keypoint(), dataset = cfg.DATASET.DATASET), 1.0,
                    "parts", dataset=cfg.DATASET.DATASET))
        if cfg.CONTRASTIVE.CUTOUT.KEYPOINT:
            self.occlusion_transforms_keypoint.append(
                RandomImageTransform(Cutout(cfg.DATASET.DATASET), 1.0, "keypoint", dataset=cfg.DATASET.DATASET))
        if cfg.CONTRASTIVE.CUTOUT.PARTS:
            self.occlusion_transforms_parts.append(
                RandomImageTransform(Cutout(cfg.DATASET.DATASET), 1.0, "parts", dataset=cfg.DATASET.DATASET))
        if cfg.CONTRASTIVE.BLURRING.KEYPOINT:
            self.occlusion_transforms_keypoint.append(
                RandomImageTransform(TargetedBlurring(cfg.DATASET.DATASET), 1.0, "keypoints",
                                     dataset=cfg.DATASET.DATASET))
        if cfg.CONTRASTIVE.BLURRING.PARTS:
            self.occlusion_transforms_parts.append(
                RandomImageTransform(TargetedBlurring(cfg.DATASET.DATASET), 1.0, "parts", dataset=cfg.DATASET.DATASET))
        if cfg.CONTRASTIVE.MULTIKEYPOINT.USE:

            multi_transforms = []
            if cfg.CONTRASTIVE.MULTIKEYPOINT.CUTOUT:
                multi_transforms.append(Cutout(cfg.DATASET.DATASET))
            if cfg.CONTRASTIVE.MULTIKEYPOINT.BLURRING:
                multi_transforms.append(TargetedBlurring(cfg.DATASET.DATASET))
            if len(multi_transforms) > 0:
                self.occlusion_transforms_keypoint.append(
                    MultiKeypointTransformation(multi_transforms, cfg.CONTRASTIVE.MULTIKEYPOINT.MAX_NUM,
                                                1.0, dataset=cfg.DATASET.DATASET))

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                    and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        keypoint_input = copy.deepcopy(input)
        keypoint_joints = copy.deepcopy(joints)
        keypoint_joints_vis = copy.deepcopy(joints_vis)
        parts_input = copy.deepcopy(input)
        parts_joints = copy.deepcopy(joints)
        parts_joints_vis = copy.deepcopy(joints_vis)

        if self.transform:
            if len(self.occlusion_transforms_keypoint) > 0:
                sample = {'image': keypoint_input, 'target': keypoint_joints, 'joints_vis': keypoint_joints_vis}
                choose_transform = random.choice(self.occlusion_transforms_keypoint)
                sample = choose_transform(sample)
                sample = self.transform(sample)
                keypoint_input = sample['image']
                keypoint_joints = sample['target']
                keypoint_joints_vis = sample['joints_vis']

            if len(self.occlusion_transforms_parts) > 0:
                sample = {'image': parts_input, 'target': parts_joints, 'joints_vis': parts_joints_vis}
                choose_transform = random.choice(self.occlusion_transforms_parts)
                sample = choose_transform(sample)
                sample = self.transform(sample)
                parts_input = sample['image']
                parts_joints = sample['target']
                parts_joints_vis = sample['joints_vis']

            # Casual image tranform
            sample = {'image': input, 'target': joints, 'joints_vis': joints_vis}
            sample = self.transform(sample)
            input = sample['image']
            joints = sample['target']
            joints_vis = sample['joints_vis']

        target, target_weight = self.generate_target(joints, joints_vis)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        input_combined = {'normal': input, 'keypoints_occluded': keypoint_input, 'parts_occluded': parts_input}
        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'parts_joints_vis': parts_joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input_combined, target, target_weight, meta
    
    
    def get_image_dir_path(self, cfg):
        prefix = 'test2017' if 'test' in cfg.DATASET.TRAIN_SET else cfg.DATASET.TRAIN_SET
        prefix = 'train2017' if 'train' in cfg.DATASET.TRAIN_SET else prefix
        prefix = 'val2017' if 'val' in cfg.DATASET.TRAIN_SET else prefix

        return os.path.join(cfg.DATASET.ROOT, 'images', prefix)    
