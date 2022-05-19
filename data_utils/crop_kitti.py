import os
from PIL import Image
root = '../VOCdevkit/VOC2012/JPEGImages_png'

img_list = [img_name for img_name in os.listdir(root)]
height = 352
width = 1216

for idx,img_name in enumerate(img_list):
    ori_img = Image.open(os.path.join(root, img_name))
    crop_x1 = int((ori_img.size[1] - height) / 2)
    crop_x2 = crop_x1 + height
    crop_y1 = int((ori_img.size[0] - width) / 2)
    crop_y2 = crop_y1 + width
    img = ori_img.crop((crop_y1, crop_x1, crop_y2, crop_x2))
    # img.save(os.path.join("../VOCdevkit/VOC2012/res",img_name))

# 对Annotations中object 坐标处理仅需要x1 - crop_x1,x2 - crop_x1;y1 - crop_y1,y2 - crop_y1;边界外特判即可(注意坐标为0、宽高小于0的情况)