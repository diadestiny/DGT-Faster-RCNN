import cv2
import numpy as np
rootDir = 'depth/'
outDir = 'result/'
def transfer_16bit_to_8bit(image_path):
    image_16bit = cv2.imread(rootDir+image_path, cv2.IMREAD_UNCHANGED)
    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)
    # image_8bit = np.array(np.rint((255.0 * (image_16bit - min_16bit)) / float(max_16bit - min_16bit)), dtype=np.uint8)
    # 或者下面一种写法
    image_8bit = np.array(np.rint(255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    print(image_16bit.dtype)
    print('16bit dynamic range: %d - %d' % (min_16bit, max_16bit))
    print(image_8bit.dtype)
    print('8bit dynamic range: %d - %d' % (np.min(image_8bit), np.max(image_8bit)))
    cv2.imwrite(outDir+image_path,image_8bit)
import os

for name in os.listdir(rootDir):
    transfer_16bit_to_8bit(name)