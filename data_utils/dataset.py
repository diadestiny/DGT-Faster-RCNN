from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree


class VOCDataSet(Dataset):
    """读取解析PASCAL VOC2007/2012数据集"""

    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt",dataset_name="kitti"):
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        if dataset_name == "kitti":
            self.root = os.path.join(voc_root, "VOCdevkit_kitti", f"VOC{year}")
            self.img_root = os.path.join(self.root, "JPEGImages")
        else:
            self.root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
            self.img_root = os.path.join(self.root, "Huxiaowei_gt")

        self.img_rain_root = os.path.join(self.root, "JPEGImages_rain_all")
        self.depth_root = os.path.join(self.root, "depth")
        self.annotations_root = os.path.join(self.root, "Annotations")
        self.dataset_name = dataset_name
        # read train.txt or val.txt file
        txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        with open(txt_path) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                             for line in read.readlines() if len(line.strip()) > 0]

        # check file
        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)
        # for xml_path in self.xml_list:
        #     assert os.path.exists(xml_path), "not found '{}' file.".format(xml_path)

        # read class_indict
        json_file = 'pascal_voc_classes_'+dataset_name+'.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        json_file = open(json_file, 'r')
        self.class_dict = json.load(json_file)
        json_file.close()

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        if self.dataset_name == "kitti":
            with open(xml_path) as fid:
                xml_str = fid.read()
        else:
            with open(xml_path.split('_rain_')[0]+".xml") as fid:
                xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_rain_path = os.path.join(self.img_rain_root, data["filename"])
        depth_path = os.path.join(self.depth_root, data["filename"])
        if self.dataset_name == "kitti":
            img_path = os.path.join(self.img_root, data["filename"])
            image_rain = Image.open(img_rain_path).convert('RGB')
        else:
            img_path = img_rain_path.replace('JPEGImages_rain_all/', 'Huxiaowei_gt/').split('_leftImg8bit')[0] + "_leftImg8bit.png"
            image_rain = Image.open(xml_path.replace('Annotations', 'JPEGImages_rain_all')[:-4] + ".png").convert('RGB')
            depth_path = depth_path.split('_leftImg8bit')[0] + "_depth_rain.png"
        image = Image.open(img_path).convert('RGB')
        depth = Image.open(depth_path)
        # r, g, b= image.split()
        # image = Image.merge("RGB", (r, g, b))
        # image = image.convert('RGB')
        boxes = []
        labels = []
        iscrowd = []
        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target, image_rain, depth= self.transforms(image, target,image_rain,depth)

        # if "alpha_0.01" in xml_path:
        #     label_index = 0
        # elif "alpha_0.02" in xml_path:
        #     label_index = 1
        # elif "alpha_0.03" in xml_path:
        #     label_index = 2

        return image, target, image_rain, depth

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        if self.dataset_name == "kitti":
            with open(xml_path) as fid:
                xml_str = fid.read()
        else:
            with open(xml_path.split('_rain_')[0] + ".xml") as fid:
                xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间
        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        if self.dataset_name == "kitti":
            with open(xml_path) as fid:
                xml_str = fid.read()
        else:
            with open(xml_path.split('_rain_')[0] + ".xml") as fid:
                xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])

            # iscrowd.append(int(obj["difficult"]))
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

import transforms
from draw_box_utils import draw_box
from PIL import Image
import json
import matplotlib.pyplot as plt
import torchvision.transforms as ts
import random

# read class_indict
# category_index = {}
# try:
#     json_file = open('./pascal_voc_classes_kitti.json', 'r')
#     class_dict = json.load(json_file)
#     category_index = {v: k for k, v in class_dict.items()}
# except Exception as e:
#     print(e)
#     exit(-1)
#
# data_transform = {
#     "train": transforms.Compose([transforms.ToTensor(),
#                                  transforms.RandomHorizontalFlip(0.5)]),
#     "val": transforms.Compose([transforms.ToTensor()])
# }
#
# # load train data set
# train_data_set = VOCDataSet("../", "2012", data_transform["train"], "train.txt")
# print(len(train_data_set))
# for index in random.sample(range(0, len(train_data_set)), k=5):
#     img, target,rain,depth = train_data_set[index]
#     img = ts.ToPILImage()(img)
#     draw_box(img,
#              target["boxes"].numpy(),
#              target["labels"].numpy(),
#              [1 for i in range(len(target["labels"].numpy()))],
#              category_index,
#              thresh=0.5,
#              line_thickness=5)
#     plt.imshow(img)
#     plt.show()
