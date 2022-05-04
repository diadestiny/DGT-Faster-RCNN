import random
from torchvision.transforms import functional as F


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target,image_rain,depth):
        for t in self.transforms:
            image, target,image_rain,depth = t(image, target,image_rain,depth)
        return image, target,image_rain,depth


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target,image_rain,depth):
        image = F.to_tensor(image)
        image_rain = F.to_tensor(image_rain)
        depth = F.to_tensor(depth)
        return image, target, image_rain, depth


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target,image_rain,depth):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            image_rain = image_rain.flip(-1)  # 水平翻转图片
            depth = depth.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target,image_rain,depth
