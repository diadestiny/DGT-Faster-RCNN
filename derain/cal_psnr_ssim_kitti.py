import os

import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from nets import basic, depth_predciton, basic_NL, DGNLNet, DensityNet, MyNet, MyNet_DGNL, MyNet_Basic, MyNet_CFT, \
    DDGN_Depth, DDGN_Basic, DDGN_Depth_CFT, DDGN_Density, DDGN_Depth_Pred, DDGN_Depth_CFT_Pred
from config import dataset_dir, dataset_dir
from misc import check_mkdir
torch.manual_seed(2019)
# torch.cuda.set_device(0)
ckpt_path = 'ckpt'
exp_name = 'kitti_DGNL'
args = {
    'snapshot': 'iter_40000_loss1_0.01177_loss2_0.00000_lr_0.000000',
    'depth_snapshot': ''
}

# mask:iter_27500_loss1_0.01083_loss2_0.00000_lr_0.000500
# density:iter_28000_loss1_0.00881_loss2_0.00000_lr_0.000500
# depth:iter_29500_loss1_0.01030_loss2_0.00000_lr_0.000150
# depth_dgnl:iter_29500_loss1_0.00864_loss2_0.00000_lr_0.000150
# base:iter_28000_loss1_0.16023_loss2_0.00000_lr_0.000169
# depth_transformer:40000、29500、27000、33000、28500、32500、10000、35000*、35500、36000*、36500*、37000、38000、38500
# 15500*、15000、16000、16500*、17500、14000、20500、21000、21500、23000、26500
# iter_16500_loss1_0.00871_loss2_0.00000_lr_0.000310

# base:iter_20000_loss1_0.01971_loss2_0.00000_lr_0.000268

transform = transforms.Compose([
    transforms.Resize([512,1024]),
    transforms.ToTensor() ])

# root = os.path.join(test_raincityscapes_path, 'JPEGImages_rain')


# root = './images'
de_rain_save_root = "results/temp2"
depth_save_root = "results/depth"

if not os.path.exists(de_rain_save_root):
    os.mkdir(de_rain_save_root)
if not os.path.exists(depth_save_root):
    os.mkdir(depth_save_root)
to_pil = transforms.ToPILImage()

net = DGNLNet().cuda()
net.eval()
if len(args['snapshot']) > 0:
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'),
                                   map_location=lambda storage, loc: storage.cuda()))
avg_time = 0
list_psnr = []
list_ssim = []
with torch.no_grad():
    input = open(os.path.join(dataset_dir, 'ImageSets', 'Main', 'val.txt'))
    gt_input = open(os.path.join(dataset_dir, 'ImageSets', 'Main', 'val.txt'))
    depth_input = open(os.path.join(dataset_dir, 'ImageSets', 'Main', 'val.txt'))
    image = [(os.path.join(dataset_dir, 'JPEGImages_rain_all', img_name.strip('\n')) + ".png") for img_name in
             input]
    gt = [(os.path.join(dataset_dir, 'JPEGImages', img_name.strip('\n')) + ".png") for img_name in
          gt_input]
    depth = [(os.path.join(dataset_dir, 'rain_mask_200', img_name.strip('\n')) + ".png") for img_name in
             depth_input]

    for idx, data in enumerate(image):
        img = Image.open(image[idx]).convert('RGB')
        gt_img = Image.open(gt[idx]).convert('RGB')
        depth_img = Image.open(depth[idx]).convert('L')
        # depth_img_L = Image.open(depth[idx].split('_rain')[0].replace('leftImg8bit', 'depth_rain') + ".png").convert('L')
        w, h = img.size
        img_var = transform(img).unsqueeze(0).cuda()
        gt_var = transform(gt_img).unsqueeze(0).cuda()
        depth_var = transform(depth_img).unsqueeze(0).cuda()
        # mask_dps = ((img_var- gt_var)[:, 0, :, :]).type(torch.cuda.FloatTensor)
        # mask_dps = mask_dps.unsqueeze(1)

        # toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
        # pic = toPIL(mask_dps)
        # pic.save('random.jpg')
        start_time = time.time()
        # , torch.Tensor([2]).cuda()
        res = net(img_var)
        # print(dps.max(1)[1] + 1)
        avg_time = avg_time + time.time() - start_time

        # torch.cuda.synchronize()
        print('predicting: %d / %d, avg_time: %.5f' % (idx + 1, len(image), avg_time / (idx + 1)))

        result = transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu()))

        # print(os.path.join(de_rain_save_root,image[idx].split('/')[-1]))
        # print(os.path.join(depth_save_root, gt[idx].split('/')[-1]))

        result.save(os.path.join(de_rain_save_root,image[idx].split('/')[-1]))
        # result.save(os.path.join(de_rain_save_root,"temp.png"))
        img_a = cv2.imread(os.path.join(de_rain_save_root,image[idx].split('/')[-1]))
        # img_a  =cv2.imread(image[idx])
        img_b = cv2.imread(gt[idx])
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        psnr_num = psnr(img_a, img_b)
        ssim_num = ssim(img_a, img_b)
        list_ssim.append(ssim_num)
        list_psnr.append(psnr_num)
        print("平均PSNR:", np.mean(list_psnr))  # ,list_psnr)
        print("平均SSIM:", np.mean(list_ssim))  # ,list_ssim)
        # if len(args['depth_snapshot']) > 0:
        #     check_mkdir(
        #         os.path.join(ckpt_path, exp_name, '(%s) prediction_%s' % (exp_name, args['depth_snapshot'])))
        #
        # img = Image.open(os.path.join(root, img_name)).convert('RGB')

        # res = net(img_var)
        # print(dps)
        # depth = transforms.Resize((h, w))(to_pil(dps.data.squeeze(0).cpu()))
        # img_name = img_name[:-4]+"_depth"+".png"
        # depth.save(os.path.join(depth_save_root,image[idx].split('/')[-1]))
        # print(idx)
    print("平均PSNR:", np.mean(list_psnr))  # ,list_psnr)
    print("平均SSIM:", np.mean(list_ssim))  # ,list_ssim)
