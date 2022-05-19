# author: Anthony
import os
import sys
import time

# cmd = 'python train_res50_fpn.py'
#
#
# def gpu_info(gpu_index):
#     gpu_status = os.popen('nvidia-smi | grep %').read().split('\n')[gpu_index].split('|')
#     power = int(gpu_status[1].split()[-3][:-1])
#     memory = int(gpu_status[2].split('/')[0].strip()[:-3])
#     return power, memory
#
#
# def narrow_setup(interval=2):
#     id = [4]
#     for gpu_id in id:
#         gpu_power, gpu_memory = gpu_info(gpu_id)
#         i = 0
#         while gpu_memory > 1000 or gpu_power > 20:  # set waiting condition
#             gpu_power, gpu_memory = gpu_info(gpu_id)
#             i = i % 5
#             symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
#             gpu = 'gpu id:%d' % gpu_id
#             gpu_power_str = 'gpu power:%d W |' % gpu_power
#             gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
#             sys.stdout.write('\r' + gpu + ' ' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
#             sys.stdout.flush()
#             time.sleep(interval)
#             i += 1
#     os.system(cmd)
#
#
# if __name__ == '__main__':
#     narrow_setup()
from shutil import copyfile

from monodepth2.utils import confidence, BilateralGrid, grid_params, BilateralSolver, bs_params

root = os.path.join("../", "VOCdevkit", "VOC2012")
img_root = os.path.join(root, "JPEGImages_RESCAN")
save = os.path.join(root, "Annotations")
yuan_xml = os.path.join(root, "Annotations2")
# for path in os.listdir(img_root):
#     name = path[:-4]
#     xml_name = name+".xml"
#     yuan = name.split("_rain")[0]+".xml"
#     copyfile(os.path.join(yuan_xml,yuan),os.path.join(save,xml_name))
#     print(os.path.join(save,xml_name))

# import xml.etree.ElementTree as ET
# for xmlpath in os.listdir(save):
#     print(xmlpath)
#     name = xmlpath[:-4]+".png"
#     doc = ET.parse(os.path.join(save,xmlpath))
#     root = doc.getroot()
#     sub1 = root.find('filename')  # 找到filename标签，
#     sub1.text = name  # 修改标签内容
#     # break
#     doc.write(os.path.join(save,xmlpath))  # 保存修改





# from lxml import etree
#
# dic = dict()
# dic["person"] = 0
# dic["rider"] = 0
# dic["car"] = 0
# dic["truck"] = 0
# dic["bus"] = 0
# dic["train"] = 0
# dic["motorcycle"] = 0
# dic["bicycle"] = 0
#
#
# for xmlpath in os.listdir(yuan_xml):
#     xmlpath = os.path.join(yuan_xml,xmlpath)
#     with open(xmlpath) as fid:
#         xml_str = fid.read()
#     xml = etree.fromstring(xml_str)
#     data = parse_xml_to_dict(xml)["annotation"]
#     if "object" not in data.keys():
#         continue
#     for obj in data["object"]:
#         class_name = obj["name"]
#         dic[class_name] = dic[class_name] + 1
#
# print(dic)
import cv2
import numpy as np
input_dir = "/data3/linkaihao/VOCdevkit/VOC2012/depth"
output_dir = "/data3/linkaihao/VOCdevkit/VOC2012/depth_solve/"
for path in os.listdir(input_dir):
    img_depth_path = os.path.join(input_dir,path)

    img_path = img_depth_path.replace('/depth/','/Huxiaowei_gt/').replace('depth_rain','leftImg8bit')
    image_rgb = cv2.imread(img_path)
    reference = image_rgb  # uint8
    im_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    con = confidence(im_gray)  # you shall try with *or* without confidence
    im_shape = reference.shape[:2]
    image_depth = cv2.imread(img_depth_path, 0)
    target = image_depth  # uint8
    grid = BilateralGrid(reference, **grid_params)
    t = target.reshape(-1, 1).astype(np.double) / (pow(2, 16) - 1)
    c = con.reshape(-1, 1).astype(np.double) / (pow(2, 16) - 1)
    output_solver = BilateralSolver(grid, bs_params).solve(t, c).reshape(im_shape)
    depth_filtersolver = np.uint16(output_solver * (pow(2, 16) - 1))
    depth_filtersolver_m = depth_filtersolver / 256.  # As meters
    cv2.imwrite(output_dir+path[:-4]+".jpg", (depth_filtersolver_m * 256.).astype(np.uint16))
    print(output_dir+path)
