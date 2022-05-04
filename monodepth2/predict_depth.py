import numpy as np
import cv2
from scipy.sparse import csr_matrix
from skimage.io import imread, imsave
from urllib3.connectionpool import xrange

# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torchvision import transforms, datasets
import monodepth2

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def simple(image_path,out_image_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_path = os.path.join("models", "mono+stereo_640x192")
    print("Loading models from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("Loading pretrained encoder")
    encoder = monodepth2.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this models was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("Loading pretrained decoder")
    depth_decoder = monodepth2.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        input_image = pil.open(image_path).convert('RGB')
        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)

        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)
        name_dest_im = 'test_image_disp.png'
        im.save(name_dest_im)
        image_depth = cv2.imread('test_image_disp.png')
        gray_img=cv2.cvtColor(image_depth,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(out_image_path,gray_img)
    print('-> Done!')


if __name__ == '__main__':
    image_path = 'test_image.jpg'
    out_image_path = 'res.png'
    simple(image_path,out_image_path)



