import os
import cv2
from config import dataset_dir

input = os.path.join(dataset_dir, 'JPEGImages_rain_new/val')
output = os.path.join(dataset_dir, 'JPEGImages_rain_new/val_resize')
for path in os.listdir(input):
    img_path = os.path.join(input,path)
    img = cv2.imread(img_path)
    x,y=img.shape[0:2]
    res_img = cv2.resize(img,(int(y*2),int(x*2)))
    cv2.imwrite(os.path.join(output,path),res_img)
    print(os.path.join(output,path))