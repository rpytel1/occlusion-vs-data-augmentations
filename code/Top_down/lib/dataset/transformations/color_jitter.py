import random

from torchvision.transforms import ColorJitter
from PIL import Image
import numpy as np


class ColorJitterWrapper(object):
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        img = sample['image']
        img = self.jitter(img)
        sample['image'] = img

        return sample

    def jitter(self, img):
        pil_image = Image.fromarray(np.uint8(img))
        magnitude = random.uniform(0, 0.05)
        tfms = ColorJitter(brightness=magnitude, contrast=magnitude, saturation=magnitude, hue=magnitude)
        pil_image = tfms(pil_image)
        img = np.array(pil_image)

        return img
