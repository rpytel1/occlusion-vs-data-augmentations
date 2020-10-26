from torchvision.transforms.functional import to_tensor, normalize


class NormalizeWrapper(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        img = sample['image']
        norm_img = normalize(img, self.mean, self.std, self.inplace)
        sample['image'] = norm_img
        return sample


class ToTensorWrapper(object):
    def __call__(self, sample):
        img = sample['image']
        img_tensor = to_tensor(img)
        sample['image'] = img_tensor
        return sample
