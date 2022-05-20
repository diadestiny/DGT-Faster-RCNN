#encoding:utf-8
import os

from nets import DDGN_Depth_CFT
from network_files.faster_rcnn_framework import Derain_FasterRCNN

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_box


def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # models = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("using {} device.".format(device))

    # create models
    model = create_model(num_classes=8+1)
    de_rain_model = DDGN_Depth_CFT()

    # load train weights
    train_weights = "./save_weights/cityscapes-20220219-39.pth"
    # train_weights = "./multi_train/model_19.pth"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)
    de_rain_model.load_state_dict(torch.load("./backbone/kitti_iter_40000_loss1_0.01115.pth"))
    myModel = Derain_FasterRCNN(FasterRCNN=model, device=device)
    myModel.to(device)
    # read class_indict
    label_json_path = 'pascal_voc_classes_kitti.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}

    dir = "/data3/linkaihao/VOCdevkit/VOC2012/JPEGImages_RESCAN/"
    for path in os.listdir(dir):
        path = dir+path
        # load image
        original_img = Image.open(path).convert("RGB")
        # original_img = Image.open("images/t3.png").convert("RGB")
        # height = 352
        # width = 1216
        # crop_x1 = int((original_img.size[1] - height) / 2)
        # crop_x2 = crop_x1 + height
        # crop_y1 = int((original_img.size[0] - width) / 2)
        # crop_y2 = crop_y1 + width
        # original_img = original_img.crop((crop_y1,crop_x1,crop_y2,crop_x2))

        # original_img = original_img.resize(size=(1242, 375))
        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        #
        # to_pil = transforms.ToPILImage()
        # h, w = 352, 1216
        # result = transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu()))
        # result.save("test_derain.png")
        # depth_res = transforms.Resize((h, w))(to_pil(depth_pred.data.squeeze(0).cpu()))
        # depth_res.save("test_depth.png")

        myModel.eval()  # 进入验证模式
        with torch.no_grad():
            # init
            t_start = time_synchronized()
            predictions = myModel(img.to(device))[0]
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")

            draw_box(original_img,
                     predict_boxes,
                     predict_classes,
                     predict_scores,
                     category_index,
                     thresh=0.5,
                     line_thickness=3)
            plt.imshow(original_img)
            plt.show()
            # 保存预测的图片结果
            # original_img.save("test_result.jpg")


if __name__ == '__main__':
    main()