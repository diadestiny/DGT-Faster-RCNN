import os
import torch
import torch.nn.functional as F
from torchsummary import summary
from torch import nn
from torch.autograd import Variable
from modules import DilatedResidualBlock, NLB, DGNL, DepthWiseDilatedResidualBlock, VIT
import torchvision.models as models
# from torchstat import stat
class DGNLNet_fast(nn.Module):
    def __init__(self, num_features=64):
        super(DGNLNet_fast, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False

        ############################################ Depth prediction network
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4, padding=2),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SELU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SELU(inplace=True)
        )

        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        )

        self.depth_pred = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        ############################################ Rain removal network

        self.head = nn.Sequential(
            # pw
            nn.Conv2d(3, 32, 1, 1, 0, 1, bias=False),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(32, 32, kernel_size=8, stride=4, padding=2, groups=32, bias=False),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(32, num_features, 1, 1, 0, 1, 1, bias=False),
        )

        self.body = nn.Sequential(
            DepthWiseDilatedResidualBlock(num_features, num_features, 1),
            # DepthWiseDilatedResidualBlock(num_features, num_features, 1),
            DepthWiseDilatedResidualBlock(num_features, num_features, 2),
            DepthWiseDilatedResidualBlock(num_features, num_features, 2),
            DepthWiseDilatedResidualBlock(num_features, num_features, 4),
            DepthWiseDilatedResidualBlock(num_features, num_features, 8),
            DepthWiseDilatedResidualBlock(num_features, num_features, 4),
            DepthWiseDilatedResidualBlock(num_features, num_features, 2),
            DepthWiseDilatedResidualBlock(num_features, num_features, 2),
            # DepthWiseDilatedResidualBlock(num_features, num_features, 1),
            DepthWiseDilatedResidualBlock(num_features, num_features, 1)
        )

        self.dgnlb = DGNL(num_features)

        self.tail = nn.Sequential(
            # dw
            nn.ConvTranspose2d(num_features, 32, kernel_size=8, stride=4, padding=2, groups=32, bias=False),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(32, 3, 1, 1, 0, 1, bias=False),
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std

        ################################## depth prediction
        d_f1 = self.conv1(x)
        d_f2 = self.conv2(d_f1)
        d_f3 = self.conv3(d_f2)
        d_f5 = self.conv5(d_f3)
        d_f8 = self.conv8(d_f5)
        d_f9 = self.conv9(d_f8 + d_f2)
        depth_pred = self.depth_pred(d_f9 + d_f1)

        ################################## rain removal

        f = self.head(x)
        f = self.body(f)
        f = self.dgnlb(f, depth_pred.detach())
        r = self.tail(f)
        x = x + r

        x = (x * self.std + self.mean).clamp(min=0, max=1)

        if self.training:
            return x, F.upsample(depth_pred, size=x.size()[2:], mode='bilinear', align_corners=True)

        return x

class DGNLNet(nn.Module):
    def __init__(self, num_features=64):
        super(DGNLNet, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False

        ############################################ Depth prediction network
        self.conv1 = nn.Sequential(
			nn.Conv2d(3, 32, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True)
		)

        self.conv2 = nn.Sequential(
			nn.Conv2d(32, 64, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=64),
			nn.SELU(inplace=True)
		)

        self.conv3 = nn.Sequential(
			nn.Conv2d(64, 128, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=128),
			nn.SELU(inplace=True)
		)

        self.conv4 = nn.Sequential(
			nn.Conv2d(128, 256, 4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=256),
			nn.SELU(inplace=True)
		)

        self.conv5 = nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=2, dilation=2),
			nn.GroupNorm(num_groups=32, num_channels=256),
			nn.SELU(inplace=True)
		)

        self.conv6 = nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=4, dilation=4),
			nn.GroupNorm(num_groups=32, num_channels=256),
			nn.SELU(inplace=True)
		)

        self.conv7 = nn.Sequential(
			nn.Conv2d(256, 256, 3, padding=2, dilation=2),
			nn.GroupNorm(num_groups=32, num_channels=256),
			nn.SELU(inplace=True)
		)

        self.conv8 = nn.Sequential(
			nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=128),
			nn.SELU(inplace=True)
		)

        self.conv9 = nn.Sequential(
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=64),
			nn.SELU(inplace=True)
		)

        self.conv10 = nn.Sequential(
			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True)
		)

        self.depth_pred = nn.Sequential(
			nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
			nn.GroupNorm(num_groups=32, num_channels=32),
			nn.SELU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.SELU(inplace=True),
			nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
			nn.Sigmoid()
		)

############################################ Rain removal network

        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0), nn.ReLU()
        )
        self.body = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 4),
            DilatedResidualBlock(num_features, 8),
            DilatedResidualBlock(num_features, 4),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 1)
        )

        self.dgnlb = DGNL(num_features)

        self.tail = nn.Sequential(
            # nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std

        ################################## depth prediction
        d_f1 = self.conv1(x)
        d_f2 = self.conv2(d_f1)
        d_f3 = self.conv3(d_f2)
        d_f4 = self.conv4(d_f3)
        d_f5 = self.conv5(d_f4)
        d_f6 = self.conv6(d_f5)
        d_f7 = self.conv7(d_f6)
        d_f8 = self.conv8(d_f7)
        d_f9 = self.conv9(d_f8 + d_f3)
        d_f10 = self.conv10(d_f9 + d_f2)
        depth_pred = self.depth_pred(d_f10 + d_f1)

        ################################## rain removal

        f = self.head(x)
        f = self.body(f)
        f = self.dgnlb(f, depth_pred.detach())
        r = self.tail(f)
        x = x + r

        x = (x * self.std + self.mean).clamp(min=0, max=1)

        if self.training:
            return x, depth_pred

        return x

class basic_NL(nn.Module):
    def __init__(self, num_features=64):
        super(basic_NL, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False

        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0), nn.ReLU()
        )
        self.body = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 4),
            DilatedResidualBlock(num_features, 8),
            DilatedResidualBlock(num_features, 4),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 1)
        )

        self.nlb = NLB(num_features)

        self.tail = nn.Sequential(
            # nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std

        f = self.head(x)
        f = self.body(f)
        f = self.nlb(f)
        r = self.tail(f)
        x = x + r

        x = (x * self.std + self.mean).clamp(min=0, max=1)
        return x

class basic(nn.Module):
    def __init__(self, num_features=64):
        super(basic, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), requires_grad=False)

        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0), nn.ReLU() #1*1卷积调整通道数
        )
        self.body = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 4),
            DilatedResidualBlock(num_features, 8),
            DilatedResidualBlock(num_features, 4),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 1)
        )

        self.tail = nn.Sequential(
            # nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )
        # 原地操作,节约显存
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std

        f = self.head(x)
        f = self.body(f)
        r = self.tail(f)
        x = x + r

        x = (x * self.std + self.mean).clamp(min=0, max=1)
        return x

class depth_predciton(nn.Module):
    def __init__(self):
        super(depth_predciton, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), requires_grad=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SELU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        )

        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SELU(inplace=True)
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        )

        self.conv10 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        )

        self.depth_pred = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.SELU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = (x - self.mean) / self.std

        d_f1 = self.conv1(x)
        d_f2 = self.conv2(d_f1)
        d_f3 = self.conv3(d_f2)
        d_f4 = self.conv4(d_f3)

        d_f5 = self.conv5(d_f4)
        d_f6 = self.conv6(d_f5)
        d_f7 = self.conv7(d_f6)

        d_f8 = self.conv8(d_f7)
        d_f9 = self.conv9(d_f8 + d_f3)
        d_f10 = self.conv10(d_f9 + d_f2)
        depth_pred = self.depth_pred(d_f10 + d_f1)

        return depth_pred

# gt_net = generator().to(device)
# summary(gt_net, (3,512,1024))

class MyNet(nn.Module):
    def __init__(self, num_features=64):
        super(MyNet, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), requires_grad=False)

        ############################################ Depth prediction network
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5)
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5)
        )

        self.conv10 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5)
        )

        self.depth_pred = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),#nn.SELU
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.dconv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.dconv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2, stride=2),
            nn.ReLU(inplace=True)
        )

        ############################################ Rain removal network

        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0), nn.ReLU()
        )
        self.body = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            # DilatedResidualBlock(num_features, 4),
            # DilatedResidualBlock(num_features, 4),
            # DilatedResidualBlock(num_features, 2),
            # DilatedResidualBlock(num_features, 1)
        )

        # self.dgnlb = DGNL(num_features)

        self.tail = nn.Sequential(
            # nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std

        ################################## depth prediction
        d_f1 = self.conv1(x)
        d_f2 = self.conv2(d_f1)
        d_f3 = self.conv3(d_f2)
        d_f4 = self.conv4(d_f3)
        d_f5 = self.conv5(d_f4)
        d_f6 = self.conv6(d_f5)
        d_f7 = self.conv7(d_f6)
        d_f8 = self.conv8(d_f7)
        d_f9 = self.conv9(d_f8 + d_f3)
        d_f10 = self.conv10(d_f9 + d_f2)
        depth_pred = self.depth_pred(d_f10 + d_f1)
        # print(depth_pred.shape)
        depth_pred1 = self.dconv1(depth_pred)
        # print(depth_pred.shape)
        depth_pred2 = self.dconv2(depth_pred1)
        # print(depth_pred.shape)

        ################################## rain removal

        f = self.head(x)
        f = self.body(f)
        f = depth_pred2*f
        # f = self.dgnlb(f, depth_pred.detach())
        r = self.tail(f)
        x = x + r
        x = (x * self.std + self.mean).clamp(min=0, max=1)
        return x,depth_pred
        # return x

class DensityNet(nn.Module):
    def __init__(self, num_features=64):
        super(DensityNet, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), requires_grad=False)

        ############################################ Depth prediction network

        haze_class = models.vgg19_bn(pretrained=True)
        self.feature = nn.Sequential(haze_class.features[0])

        for i in range(1, 3):
            self.feature.add_module(str(i), haze_class.features[i])
            print(haze_class.features[i])

        self.conv16 = nn.Conv2d(64, 24, kernel_size=3, stride=1, padding=1)  # 1mm
        self.dense_classifier = nn.Linear(207600, 512)
        self.dense_classifier1 = nn.Linear(512, 3)

        self.conv9 = nn.Conv2d(128, 64, 1, 1)
        ############################################ Rain removal network

        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0), nn.ReLU()
        )
        self.body = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            # DilatedResidualBlock(num_features, 4),
            # DilatedResidualBlock(num_features, 4),
            # DilatedResidualBlock(num_features, 2),
            # DilatedResidualBlock(num_features, 1)
        )

        # self.dgnlb = DGNL(num_features)

        self.dconv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1,stride=1),
            nn.ReLU(inplace=True),
        )

        self.dconv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.tail = nn.Sequential(
            # nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std

        ################################## depth prediction
        feature = self.feature(x)
        feature = self.conv16(feature)
        out = F.relu(feature, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(out.size(0), -1)

        out = F.relu(self.dense_classifier(out))
        out = (self.dense_classifier1(out))

        f_out = out.max(1)[1] + 1
        f_out = f_out.unsqueeze(dim=1)
        f_out = f_out.unsqueeze(dim=2)
        f_out = f_out.unsqueeze(dim=3)

        ################################## rain removal

        f = self.head(x)
        f = self.body(f)

        f_out = f_out.expand_as(f)
        f = torch.cat([f, f_out], dim=1)

        f = self.conv9(f)
        # f = self.dgnlb(f, depth_pred.detach())
        r = self.tail(f)
        x = x + r
        x = (x * self.std + self.mean).clamp(min=0, max=1)

        return x,out

class MyNet_Basic(nn.Module):
    def __init__(self, num_features=64):
        super(MyNet_Basic, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), requires_grad=False)

        ############################################ Rain removal network
        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0), nn.ReLU()
        )
        self.body = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            # DilatedResidualBlock(num_features, 4),
            # DilatedResidualBlock(num_features, 4),
            # DilatedResidualBlock(num_features, 2),
            # DilatedResidualBlock(num_features, 1)
        )

        self.tail = nn.Sequential(
            # nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std
        ################################## rain removal
        f = self.head(x)
        f = self.body(f)
        r = self.tail(f)
        x = x + r
        x = (x * self.std + self.mean).clamp(min=0, max=1)
        return x

class MyNet_DGNL(nn.Module):
    def __init__(self, num_features=64):
        super(MyNet_DGNL, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), requires_grad=False)

        ############################################ Depth prediction network
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5)
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5)
        )

        self.conv10 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5)
        )

        self.depth_pred = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        ############################################ Rain removal network

        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0), nn.ReLU()
        )
        self.body = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 4),
            DilatedResidualBlock(num_features, 4),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 1)
        )

        self.dgnlb = DGNL(num_features)

        # self.dconv1 = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, padding=1,stride=1),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.dconv2 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2, stride=2),
        #     nn.ReLU(inplace=True)
        # )

        self.tail = nn.Sequential(
            # nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std

        ################################## depth prediction
        d_f1 = self.conv1(x)
        d_f2 = self.conv2(d_f1)
        d_f3 = self.conv3(d_f2)
        d_f4 = self.conv4(d_f3)
        d_f5 = self.conv5(d_f4)
        d_f6 = self.conv6(d_f5)
        d_f7 = self.conv7(d_f6)
        d_f8 = self.conv8(d_f7)
        d_f9 = self.conv9(d_f8 + d_f3)
        d_f10 = self.conv10(d_f9 + d_f2)
        depth_pred = self.depth_pred(d_f10 + d_f1)
        # print(depth_pred.shape)
        # depth_pred1 = self.dconv1(depth_pred)
        # print(depth_pred.shape)
        # depth_pred2 = self.dconv2(depth_pred1)
        # print(depth_pred.shape)

        ################################## rain removal

        f = self.head(x)
        f = self.body(f)
        # f = depth_pred2*f
        f = self.dgnlb(f, depth_pred.detach())
        r = self.tail(f)
        x = x + r
        x = (x * self.std + self.mean).clamp(min=0, max=1)
        return x,depth_pred

class MyNet_CFT(nn.Module):
    def __init__(self, num_features=64):
        super(MyNet_CFT, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), requires_grad=False)

        ############################################ Depth prediction network
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5)
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5)
        )

        self.conv10 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5)
        )

        self.dconv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.dconv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2, dilation=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.depth_pred = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        # self.cft = VIT(img_h=176,img_w=608,in_c=64,patch_size=16,embed_dim=768,depth=8,num_heads=4)
        self.cft = VIT(d_model=num_features)
        ############################################ Rain removal network

        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0), nn.ReLU()
        )
        self.body = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            # DilatedResidualBlock(num_features, 4),
            # DilatedResidualBlock(num_features, 4),
            # DilatedResidualBlock(num_features, 2),
            # DilatedResidualBlock(num_features, 1)
        )

        self.tail = nn.Sequential(
            # nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std

        ################################## depth prediction
        d_f1 = self.conv1(x)
        d_f2 = self.conv2(d_f1)
        d_f3 = self.conv3(d_f2)
        d_f4 = self.conv4(d_f3)
        d_f5 = self.conv5(d_f4)
        d_f6 = self.conv6(d_f5)
        d_f7 = self.conv7(d_f6)
        d_f8 = self.conv8(d_f7)
        d_f9 = self.conv9(d_f8 + d_f3)
        d_f10 = self.conv10(d_f9 + d_f2)
        depth_pred = self.depth_pred(d_f10 + d_f1)
        depth_pred1 = self.dconv1(depth_pred)
        depth_pred2 = self.dconv2(depth_pred1)

        ################################## rain removal

        f = self.head(x)
        f = self.body(f)
        f = self.cft(f, depth_pred2.detach())
        r = self.tail(f)
        x = x + r
        x = (x * self.std + self.mean).clamp(min=0, max=1)
        return x,depth_pred

class DDGN_Basic(nn.Module):
    def __init__(self, num_features=64):
        super(DDGN_Basic, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), requires_grad=False)

        ############################################ Rain removal network
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0)
        )

        self.conv_DBR = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 1)
        )

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        )
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std
        ################################## rain removal
        f = self.conv0(x)
        f = self.conv_DBR(f)
        r = self.conv1(f)
        x = x + r
        x = (x * self.std + self.mean).clamp(min=0, max=1)
        return x

class DDGN_Depth(nn.Module):
    def __init__(self, num_features=64):
        super(DDGN_Depth, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), requires_grad=False)

        ############################################ Rain removal network
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0)
        )
        self.conv_dps = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1),nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2, dilation=2),
        )
        self.conv_DBR = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 1)
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        )
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x,dps):
        x = (x - self.mean) / self.std
        # tdp = torch.cat([dps, dps, dps], dim=1)
        # tdp = (dps - self.mean) / self.std
        f = self.conv0(x)
        dps = self.conv_dps(dps)
        f = self.conv_DBR(f)
        r = self.conv1(f*dps)
        x = x + r
        x = (x * self.std + self.mean).clamp(min=0, max=1)
        return x

class DDGN_Depth_CFT(nn.Module):
    def __init__(self, num_features=64):
        super(DDGN_Depth_CFT, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), requires_grad=False)

        ############################################ Rain removal network
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0)
        )

        self.conv_dps = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2, dilation=2)
        )
        self.cft = VIT(d_model=num_features)

        self.conv_DBR = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 1)
        )

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x,dps):
        x = (x - self.mean) / self.std
        ################################## rain removal
        f = self.conv0(x)
        f = self.conv_DBR(f)
        dps = self.conv_dps(dps)
        f = self.cft(f, dps.detach())
        # f = F.softmax(f * dps,dim=2)
        r = self.conv1(f)
        x = x + r
        x = (x * self.std + self.mean).clamp(min=0, max=1)
        return x

class DDGN_Density(nn.Module):
    def __init__(self, num_features=64):
        super(DDGN_Density, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), requires_grad=False)

        ############################################ Rain removal network

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0)
        )

        self.conv9 = nn.Conv2d(128, 64, 1, 1)

        self.conv_DBR = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 1)
        )

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        )
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x, label):
        x = (x - self.mean) / self.std
        ################################## density
        f_out = label + 1
        f_out = f_out.unsqueeze(dim=1)
        f_out = f_out.unsqueeze(dim=2)
        f_out = f_out.unsqueeze(dim=3)
        ################################## rain removal
        f = self.conv0(x)
        f = self.conv_DBR(f)
        f_out = f_out.expand_as(f)
        f = torch.cat([f, f_out], dim=1)
        f = self.conv9(f)
        r = self.conv1(f)
        x = x + r
        x = (x * self.std + self.mean).clamp(min=0, max=1)
        return x

class DDGN_Depth_Pred(nn.Module):
    def __init__(self, num_features=64):
        super(DDGN_Depth_Pred, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), requires_grad=False)

        ############################################ Rain removal network
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0)
        )

        self.conv_dps = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2, dilation=2)
        )

        self.conv_DBR = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 1)
        )

        self.conv_tail = nn.Sequential(
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        )

        ############################################ Depth prediction network
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv10 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.depth_pred = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std

        d_f1 = self.conv1(x)
        d_f2 = self.conv2(d_f1)
        d_f3 = self.conv3(d_f2)
        d_f4 = self.conv4(d_f3)
        d_f5 = self.conv5(d_f4)
        d_f6 = self.conv6(d_f5)
        d_f7 = self.conv7(d_f6)
        d_f8 = self.conv8(d_f7)
        d_f9 = self.conv9(d_f8 + d_f3)
        d_f10 = self.conv10(d_f9 + d_f2)
        depth_pred = self.depth_pred(d_f10 + d_f1)

        ################################## rain removal
        f = self.conv0(x)
        f = self.conv_DBR(f)
        dps = self.conv_dps(depth_pred)
        r = self.conv_tail(f*dps)
        x = x + r
        x = (x * self.std + self.mean).clamp(min=0, max=1)
        return x,depth_pred

class DDGN_Depth_CFT_Pred(nn.Module):
    def __init__(self, num_features=64):
        super(DDGN_Depth_CFT_Pred, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), requires_grad=False)

        ############################################ Rain removal network
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0)
        )

        self.conv_dps = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2, dilation=2)
        )
        self.cft = VIT(d_model=num_features)
        self.conv_DBR = nn.Sequential(
            DilatedResidualBlock(num_features, 1),
            DilatedResidualBlock(num_features, 2),
            DilatedResidualBlock(num_features, 1)
        )

        self.conv_tail = nn.Sequential(
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        )

        ############################################ Depth prediction network
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv10 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.depth_pred = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std

        d_f1 = self.conv1(x)
        d_f2 = self.conv2(d_f1)
        d_f3 = self.conv3(d_f2)
        d_f4 = self.conv4(d_f3)
        d_f5 = self.conv5(d_f4)
        d_f6 = self.conv6(d_f5)
        d_f7 = self.conv7(d_f6)
        d_f8 = self.conv8(d_f7)
        d_f9 = self.conv9(d_f8 + d_f3)
        d_f10 = self.conv10(d_f9 + d_f2)
        depth_pred = self.depth_pred(d_f10 + d_f1)

        ################################## rain removal
        f = self.conv0(x)
        f = self.conv_DBR(f)
        dps = self.conv_dps(depth_pred)
        en_f = self.cft(f, dps.detach())
        r = self.conv_tail(en_f)
        x = x + r
        x = (x * self.std + self.mean).clamp(min=0, max=1)
        return x,depth_pred

# os.environ["CUDA_VISIBLE_DEVICES"] = '4'
# device = torch.device("cuda:0")

class Prior_Prediction_Net(nn.Module):
    def __init__(self):
        super(Prior_Prediction_Net, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1), requires_grad=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv10 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv_pred = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
        )
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        x = (x - self.mean) / self.std
        d_f1 = self.conv1(x)
        d_f2 = self.conv2(d_f1)
        d_f3 = self.conv3(d_f2)
        d_f4 = self.conv4(d_f3)
        d_f5 = self.conv5(d_f4)
        d_f6 = self.conv6(d_f5)
        d_f7 = self.conv7(d_f6)
        d_f8 = self.conv8(d_f7)
        d_f9 = self.conv9(d_f8 + d_f3)
        d_f10 = self.conv10(d_f9 + d_f2)
        conv_pred = self.conv_pred(d_f10 + d_f1)
        # conv_pred = (conv_pred * self.std + self.mean).clamp(min=0, max=1)
        return conv_pred

# from thop import profile
# from thop import clever_format
# input = torch.randn(1, 3, 512, 1024)
# dps = torch.randn(1,1,512,1024)
# model = DDGN_Depth_CFT()
# # # stat(model,(3,512,1024))
# flops, params = profile(model, inputs=(input.cpu(),dps.cpu()))
# flops, params = clever_format([flops, params], "%.3f")
# print('params:',params,'flops:',flops)
