import random

from lib.dataset.transformations.transformation_util import extract_keypoints


class MultiKeypointTransformation(object):

    def __init__(self, transforms, num_keypoints, probability, dataset):
        self.transforms = transforms
        self.num_keypoints = num_keypoints
        self.probability = probability
        self.dataset = dataset

    def __call__(self, sample):

        keypoints = list(extract_keypoints(sample['target'], self.dataset)[0].keys())

        num_iter = min(len(keypoints), self.num_keypoints)
        for i in range(num_iter):
            chance = random.random()
            if chance < self.probability:
                keypoint_name = random.choice(keypoints)
                keypoints.remove(keypoint_name)
                chosen_transform = random.choice(self.transforms)
                chosen_transform.part = keypoint_name
                sample = chosen_transform(sample)

        return sample
