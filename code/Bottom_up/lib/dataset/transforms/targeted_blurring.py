from .image_transform import ImageTransform
from .transformation_util import extract_keypoints
from .cutout_util import remove_annotation, blurr_img
from .keypoints_constants import part_mapping


class TargetedBlurring(ImageTransform):
    def __init__(self, dataset, remove_anns=False, part="nose", width=6):
        super().__init__(part)
        self.remove_anns = remove_anns
        self.dataset = dataset
        self.width = width

    def __call__(self, sample):
        img = sample['image']
        keypoint_list = extract_keypoints(sample['target'], self.dataset)
        img, center_pos, widths = blurr_img(img, keypoint_list, self.part, self.dataset, self.width)

        sample['image'] = img

        if self.part in part_mapping[self.dataset].keys() and self.remove_anns:
            keypoints_arr = sample['target']
            joints_vis = sample['joints_vis']
            keypoints_arr, joints_vis = remove_annotation(keypoints_arr, joints_vis, center_pos, widths)
            sample['target'] = keypoints_arr
            sample['joints_vis'] = joints_vis

        return sample
