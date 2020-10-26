import random

from lib.dataset.transformations.transformation_util import extract_keypoints
from lib.utils.keypoints_constants import part_mapping, advanced_parts


class RandomImageTransform(object):
    def __init__(self, image_transform, probabilty, mode, dataset):
        self.transform = image_transform
        self.probability = probabilty
        self.mode = mode
        self.dataset = dataset

    def __call__(self, sample):
        chance = random.random()

        if chance < self.probability:
            self.transform.part = self.choose_part(sample)
            sample = self.transform(sample)

        return sample

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
