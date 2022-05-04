import os
import os.path
import numpy as np
import torch.utils.data as data
from PIL import Image


def make_dataset(root, is_train,train_str):

    if is_train:
        input = open(os.path.join(root, 'ImageSets','Main','train.txt'))
        gt_input = open(os.path.join(root, 'ImageSets', 'Main', 'train.txt'))
        depth_input = open(os.path.join(root, 'ImageSets', 'Main', 'train.txt'))
        image = [(os.path.join(root, 'JPEGImages_rain_all', img_name.strip('\n'))) for img_name in
                 input]
        gt = [(os.path.join(root, 'JPEGImages', img_name.strip('\n'))) for img_name in
                 gt_input]
        if train_str == "mask":
            depth = [(os.path.join(root, 'rain_mask_200', img_name.strip('\n'))) for img_name in
                  depth_input]
            print("mask")
        else:
            depth = [(os.path.join(root, 'depth', img_name.strip('\n'))) for img_name in
                     depth_input]
            print("depth")
        input.close()

        return [[image[i]+".png", gt[i]+".png", depth[i]+".png"]for i in range(len(image))]

    else:
        input = open(os.path.join(root, 'ImageSets','Main','val.txt'))
        gt_input = open(os.path.join(root, 'ImageSets', 'Main', 'val.txt'))
        depth_input = open(os.path.join(root, 'ImageSets', 'Main', 'val.txt'))
        image = [(os.path.join(root,'JPEGImages_rain_all', img_name.strip('\n'))) for img_name in
                 input]
        gt = [(os.path.join(root, 'JPEGImages', img_name.strip('\n'))) for img_name in
              gt_input]

        if train_str == "mask":
            depth = [(os.path.join(root, 'rain_mask_200', img_name.strip('\n'))) for img_name in
                     depth_input]
        else:
            depth = [(os.path.join(root,'depth', img_name.strip('\n'))) for img_name in
                     depth_input]

        input.close()
        return [[image[i] + ".png", gt[i] + ".png", depth[i] + ".png"] for i in range(len(image))]


class ImageFolder_Density(data.Dataset):
    def __init__(self, root, triple_transform=None, transform=None, target_transform=None, is_train=True):
        self.root = root
        self.imgs = make_dataset(root, is_train,"_")
        self.triple_transform = triple_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        index_folder = np.random.randint(1,4)
        img_path, gt_path, _ = self.imgs[index]
        label = 0
        if index_folder == 1:
            if "alpha_0.02_beta_0.01_dropsize_0.005" in img_path:
                img_path = img_path.replace('alpha_0.02_beta_0.01_dropsize_0.005',
                                                'alpha_0.01_beta_0.005_dropsize_0.01')
            elif "alpha_0.03_beta_0.015_dropsize_0.002" in img_path:
                img_path = img_path.replace('alpha_0.03_beta_0.015_dropsize_0.002',
                                                'alpha_0.01_beta_0.005_dropsize_0.01')
            label = 1
        elif index_folder == 2:
            # img_path = img_path.replace("JPEGImages_rain_all","image_rain_100")
            if "alpha_0.01_beta_0.005_dropsize_0.01" in img_path:
                img_path = img_path.replace('alpha_0.01_beta_0.005_dropsize_0.01',
                                                'alpha_0.02_beta_0.01_dropsize_0.005')
            elif "alpha_0.03_beta_0.015_dropsize_0.002" in img_path:
                img_path = img_path.replace('alpha_0.03_beta_0.015_dropsize_0.002',
                                                'alpha_0.02_beta_0.01_dropsize_0.005')
            label = 2
        elif index_folder == 3:
            if "alpha_0.01_beta_0.005_dropsize_0.01" in img_path:
                img_path = img_path.replace('alpha_0.01_beta_0.005_dropsize_0.01',
                                                'alpha_0.03_beta_0.015_dropsize_0.002')
            elif "alpha_0.02_beta_0.01_dropsize_0.005" in img_path:
                img_path = img_path.replace('alpha_0.02_beta_0.01_dropsize_0.005',
                                                'alpha_0.03_beta_0.015_dropsize_0.002')
            label = 3
        # print(img_path)
        # print(label)
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('RGB')
        # depth = Image.open(depth_path)
        if self.triple_transform is not None:
            img, target = self.triple_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            # depth = self.target_transform(depth)

        return img, target, label-1

    def __len__(self):
        return len(self.imgs)


class ImageFolder(data.Dataset):
    def __init__(self, root, triple_transform=None, transform=None, target_transform=None, is_train=True,train_str="depth"):
        self.root = root
        self.imgs = make_dataset(root, is_train,train_str)
        self.triple_transform = triple_transform
        self.transform = transform
        self.target_transform = target_transform
        self.train_str = train_str

    def __getitem__(self, index):
        img_path, gt_path, depth_path = self.imgs[index]
        #print(img_path)
        #print(gt_path)
        #print(depth_path)
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('RGB')
        if self.train_str == "mask":
            depth = Image.open(img_path).convert('L')
        else:
            depth = Image.open(depth_path)
        if self.triple_transform is not None:
            img, target, depth = self.triple_transform(img, target, depth)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            depth = self.target_transform(depth)
        return img, target, depth

    def __len__(self):
        return len(self.imgs)
