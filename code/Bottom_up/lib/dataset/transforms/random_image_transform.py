import random

from .transformation_util import extract_keypoints
from .keypoints_constants import part_mapping, advanced_parts
import numpy as np

class RandomImageTransform(object):
    def __init__(self, image_transform, probabilty, mode, dataset):
        self.transform = image_transform
        self.probability = probabilty
        self.mode = mode
        self.dataset = dataset

    def __call__(self, image,mask,joints):
        num_instances, _, _ = joints[1].shape
        for i in range(num_instances):
            chance = random.random()
            if chance < self.probability:
                instance_joints = joints[1][i,:,:]
                sample = {'image':image, 'target':instance_joints,'joints_vis':[]}            
                self.transform.part = self.choose_part(sample)
                sample = self.transform(sample)
                image = sample['image']
                joints[1][i,:,:] = sample['target']
                

        return image, mask, joints

    def choose_part(self, sample):
        keypoint_name = ""
        keypoints = list(extract_keypoints(sample['target'], self.dataset)[0].keys())
        if len(keypoints) > 0:
            if self.mode == "all":
                keypoint_name = random.choice(keypoints + list(part_mapping[self.dataset].keys()))
            elif self.mode == "keypoint":
                keypoint_name = random.choice(keypoints)
            else:  # Fallback to parts
                parts = list(part_mapping[self.dataset].keys())
                filtered_parts = [part for part in parts if part not in (list(advanced_parts[self.dataset].keys()))]
                keypoint_name = random.choice(filtered_parts)

        return keypoint_name
