import json

from lib.dataset.transformations.image_transform import ImageTransform
from lib.dataset.transformations.transformation_util import extract_keypoints
from lib.utils.cutout_util import remove_annotation, extract_json, get_cords_with_width, get_xy
import random
import copy
import cv2
import os

from lib.utils.keypoints_constants import part_mapping, keypoint_names


class TargetedCutmix(ImageTransform):
    parts = ["head", "left_arm", "right_arm", "left_leg", "right_leg", "corpus"]

    def __init__(self, image_path, annotation_path, dataset, remove_anns=True, part="head"):
        super().__init__(part)
        self.image_path = image_path
        self.remove_anns = remove_anns
        self.dataset = dataset
        self.anns = self.load_annotation(annotation_path)
        

    def __call__(self, sample):
        img = sample['image']
        keypoint_list = extract_keypoints(sample['target'], self.dataset)
        self.part = random.choice(self.parts)
        if self.check_if_has_part(keypoint_list[0], self.part):

            img, center_pos, widths = self.perform_cutmix(img, keypoint_list)
            sample['image'] = img

            if self.part in part_mapping[self.dataset].keys() and self.remove_anns:
                keypoints_arr = sample['target']
                joints_vis = sample['joints_vis']
                keypoints_arr, joints_vis = remove_annotation(keypoints_arr, joints_vis, center_pos, widths)
                sample['target'] = keypoints_arr
                sample['joints_vis'] = joints_vis

        return sample

    def perform_cutmix(self, img, keypoint_list):

        for keypoints in keypoint_list:
            other_part = self.get_other_part()
            other_img, key_dict = self.get_other_image(other_part)

            center, widths = get_cords_with_width(keypoints, self.part, self.dataset)
            x1, x2, y1, y2 = get_xy(center, widths, img.shape[:-1])

            dim = img[y1:y2, x1:x2].shape[:-1]
            dim = (dim[1], dim[0])
            part_img = self.get_part(other_img, other_part)
            if dim[0] > 0 and dim[1] > 0 and part_img.shape[0] > 0 and part_img.shape[1] > 0:
                part_img_resized = cv2.resize(part_img, dim, interpolation=cv2.INTER_AREA)
                img[y1:y2, x1:x2] = part_img_resized
                return img, center, widths
        return img, (0, 0), (0, 0)

    def get_part(self, imgfile, part):
        img = cv2.imread(os.path.join(self.image_path, imgfile))
        keypoints = self.anns[imgfile][0]

        center, widths = get_cords_with_width(keypoints, part, self.dataset)
        x1, x2, y1, y2 = get_xy(center, widths, img.shape[:-1])
        return img[y1:y2, x1:x2]

    def load_annotation(self, annotation_path):
        with open(annotation_path) as json_file:
            annotations_val = json.load(json_file)
        return extract_json(annotations_val, self.dataset)

    def get_other_part(self):
        parts_copy = copy.deepcopy(self.parts)
        parts_copy.remove(self.part)
        return random.choice(parts_copy)

    def get_other_image(self, other_part):
        while True:
            key = random.choice(list(self.anns.keys()))
            if len(self.anns[key]) > 0:
                chosen_annotation = self.anns[key][0]
                if self.check_if_has_part(chosen_annotation, other_part):
                    return key, chosen_annotation

    def check_if_has_part(self, keypoint_list, part):
        for elem in part_mapping[self.dataset][part]:
            if keypoint_names[self.dataset][elem] not in keypoint_list.keys():
                return False

        return True
