import os
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
import numpy as np

class DGNL(nn.Module):
    def __init__(self, in_channels):
        super(DGNL, self).__init__()

        self.eps = 1e-6
        self.sigma_pow2 = 100

        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

        # [2, 64, 176, 608]->[2, 64, 44, 152]
        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)

    def forward(self, x, depth_map):
        n, c, h, w = x.size()
        x_down = self.down(x)
        # print(x.shape)
		# [n, (h / 8) * (w / 8), c / 2]
        g = F.max_pool2d(self.g(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)
        # print(g.shape)
        ### appearance relation map
        # [n, (h / 4) * (w / 4), c / 2]
        theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)
        # print(theta.shape)
        # [n, c / 2, (h / 8) * (w / 8)]
        phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)
        # print(phi.shape)
		# [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
        Ra = F.softmax(torch.bmm(theta, phi), 2)


        ### depth relation map
        depth1 = F.interpolate(depth_map, size=[int(h / 4), int(w / 4)], mode='bilinear', align_corners = True).view(n, 1, int(h / 4)*int(w / 4)).transpose(1,2)
        depth2 = F.interpolate(depth_map, size=[int(h / 8), int(w / 8)], mode='bilinear', align_corners = True).view(n, 1, int(h / 8)*int(w / 8))

        # n, (h / 4) * (w / 4), (h / 8) * (w / 8)
        depth1_expand = depth1.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        depth2_expand = depth2.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))

        Rd = torch.min(depth1_expand / (depth2_expand + self.eps), depth2_expand / (depth1_expand + self.eps))
        Rd = F.softmax(Rd, 2)
        # normalization: depth relation map [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
        # Rd = Rd / (torch.sum(Rd, 2).view(n, int(h / 4) * int(w / 4), 1) + self.eps)

        # ### position relation map
        # position_h = torch.Tensor(range(h)).cuda().view(h, 1).expand(h, w)
        # position_w = torch.Tensor(range(w)).cuda().view(1, w).expand(h, w)
		#
        # position_h1 = F.interpolate(position_h.unsqueeze(0).unsqueeze(0), size=[int(h / 4), int(w / 4)], mode='bilinear', align_corners=True).view(1, 1, int(h / 4) * int(w / 4)).transpose(1,2)
        # position_h2 = F.interpolate(position_h.unsqueeze(0).unsqueeze(0), size=[int(h / 8), int(w / 8)], mode='bilinear', align_corners=True).view(1, 1, int(h / 8) * int(w / 8))
        # position_h1_expand = position_h1.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        # position_h2_expand = position_h2.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        # h_distance = (position_h1_expand - position_h2_expand).pow(2)
		#
        # position_w1 = F.interpolate(position_w.unsqueeze(0).unsqueeze(0), size=[int(h / 4), int(w / 4)], mode='bilinear', align_corners=True).view(1, 1, int(h / 4) * int(w / 4)).transpose(1, 2)
        # position_w2 = F.interpolate(position_w.unsqueeze(0).unsqueeze(0), size=[int(h / 8), int(w / 8)], mode='bilinear', align_corners=True).view(1, 1, int(h / 8) * int(w / 8))
        # position_w1_expand = position_w1.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        # position_w2_expand = position_w2.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        # w_distance = (position_w1_expand - position_w2_expand).pow(2)
		#
        # Rp = 1 / (2 * 3.14159265 * self.sigma_pow2) * torch.exp(-0.5 * (h_distance / self.sigma_pow2 + w_distance / self.sigma_pow2))
		#
        # Rp = Rp / (torch.sum(Rp, 2).view(n, int(h / 4) * int(w / 4), 1) + self.eps)


        ### overal relation map
        #S = F.softmax(Ra * Rd * Rp, 2)
        # S = self.cft(Ra,Rd)
        S = F.softmax(Ra * Rd, 2)
        # [n, c / 2, h / 4, w / 4]
        y = torch.bmm(S, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))

        return x + F.interpolate(self.z(y), size=x.size()[2:], mode='bilinear', align_corners = True)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_h=224,img_w = 224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_h, img_w)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match models ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class VIT(nn.Module):
    """  the full GPT language models, with a context size of block_size """

    def __init__(self, patch_size=16,embed_dim=768,img_h=224,img_w=224, in_c=3, num_classes=1000,
                depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = embed_layer(img_h=img_h, img_w=img_w, in_c=in_c,patch_size=patch_size, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * self.num_patches , embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # transformer
        self.trans_blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        # decoder head

        self.norm = norm_layer(embed_dim)
        # regularization
        self.pos_drop = nn.Dropout(p=drop_ratio)
        # self.conv_t=  nn.Sequential(nn.ConvTranspose2d(embed_dim, in_c, kernel_size=patch_size, stride=patch_size))
        self.conv_t = nn.Sequential(nn.Conv2d(embed_dim, in_c, kernel_size=1, stride=1,padding=1))
        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x1, x2):
        # assert x1.shape == x2.shape
        bs, c, h, w = x1.shape
        rgb_fea = self.patch_embed(x1)  # [B, 196, 768]
        ir_fea = self.patch_embed(x2)
        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        token_embeddings = torch.cat([rgb_fea, ir_fea], dim=1)  # concat
        # transformer
        x = self.pos_drop(self.pos_emb + token_embeddings)  # sum positional embedding and token  (B, 2*N, embed_dim)
        x = self.trans_blocks(x)  # (B, 2*N, embed_dim)
        x = self.norm(x)
        x = x.view(bs, 2, self.num_patches, self.embed_dim)
        x = x.permute(0, 1, 3, 2)  # dim:(B, 2, C, N)
        x = x.reshape(bs, 2, self.embed_dim, h//self.patch_size,w//self.patch_size) # dim:(B, 2*C, N)

        # # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.embed_dim, h//self.patch_size,w//self.patch_size)
        out = self.conv_t(rgb_fea_out)
        # ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        # rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear',align_corners=True)
        # ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear',align_corners=True)
        # return x1+F.interpolate(self.conv_t(x), size=x1.size()[2:], mode='bilinear', align_corners=True)
        return x1+F.interpolate(out, size=x1.size()[2:], mode='bilinear', align_corners=True)

class NLB(nn.Module):
    def __init__(self, in_channels):
        super(NLB, self).__init__()
        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.size()
        x_down = self.down(x)

        # [n, (h / 4) * (w / 4), c / 2]
        theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)
        # [n, c / 2, (h / 8) * (w / 8)]
        phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)
        # [n, (h / 8) * (w / 8), c / 2]
        g = F.max_pool2d(self.g(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)
        # [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
        f = F.softmax(torch.bmm(theta, phi), 2)
        # [n, c / 2, h / 4, w / 4]
        y = torch.bmm(f, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))

        return x + F.interpolate(self.z(y), size=x.size()[2:], mode='bilinear', align_corners=True)


class DepthWiseDilatedResidualBlock(nn.Module):
    def __init__(self, reduced_channels, channels, dilation):
        super(DepthWiseDilatedResidualBlock, self).__init__()
        self.conv0 = nn.Sequential(

		    # pw
		    nn.Conv2d(channels, channels * 2, 1, 1, 0, 1, bias=False),
			nn.ReLU6(inplace=True),
		    # dw
		    nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=dilation, dilation=dilation, groups=channels, bias=False),
		    nn.ReLU6(inplace=True),
		    # pw-linear
		    nn.Conv2d(channels*2, channels, 1, 1, 0, 1, 1, bias=False)
        )

        self.conv1 = nn.Sequential(
			# pw
			# nn.Conv2d(channels, channels * 2, 1, 1, 0, 1, bias=False),
			# nn.ReLU6(inplace=True),
			# dw
			nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, groups=channels,
					  bias=False),
			nn.ReLU6(inplace=True),
			# pw-linear
			nn.Conv2d(channels, channels, 1, 1, 0, 1, 1, bias=False)
		)


    def forward(self, x):
        res = self.conv1(self.conv0(x))
        return res + x


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super(DilatedResidualBlock, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation), nn.ReLU()
        )
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        return x + conv1


class SpatialRNN(nn.Module):
	"""
	SpatialRNN models for one direction only
	"""
	def __init__(self, alpha = 1.0, channel_num = 1, direction = "right"):
		super(SpatialRNN, self).__init__()
		self.alpha = nn.Parameter(torch.Tensor([alpha] * channel_num))
		self.direction = direction

	def __getitem__(self, item):
		return self.alpha[item]

	def __len__(self):
		return len(self.alpha)


	def forward(self, x):
		"""
		:param x: (N,C,H,W)
		:return:
		"""
		height = x.size(2)
		weight = x.size(3)
		x_out = []

		# from left to right
		if self.direction == "right":
			x_out = [x[:, :, :, 0].clamp(min=0)]

			for i in range(1, weight):
				temp = (self.alpha.unsqueeze(1) * x_out[i - 1] + x[:, :, :, i]).clamp(min=0)
				x_out.append(temp)  # a list of tensor

			return torch.stack(x_out, 3)  # merge into one tensor

		# from right to left
		elif self.direction == "left":
			x_out = [x[:, :, :, -1].clamp(min=0)]

			for i in range(1, weight):
				temp = (self.alpha.unsqueeze(1) * x_out[i - 1] + x[:, :, :, -i - 1]).clamp(min=0)
				x_out.append(temp)

			x_out.reverse()
			return torch.stack(x_out, 3)

		# from up to down
		elif self.direction == "down":
			x_out = [x[:, :, 0, :].clamp(min=0)]

			for i in range(1, height):
				temp = (self.alpha.unsqueeze(1) * x_out[i - 1] + x[:, :, i, :]).clamp(min=0)
				x_out.append(temp)

			return torch.stack(x_out, 2)

		# from down to up
		elif self.direction == "up":
			x_out = [x[:, :, -1, :].clamp(min=0)]

			for i in range(1, height):
				temp = (self.alpha.unsqueeze(1) * x_out[i - 1] + x[:, :, -i - 1, :]).clamp(min=0)
				x_out.append(temp)

			x_out.reverse()
			return torch.stack(x_out, 2)

		else:
			print("Invalid direction in SpatialRNN!")
			return KeyError



class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class NLB(nn.Module):
    def __init__(self, in_channels):
        super(NLB, self).__init__()
        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.size()
        x_down = self.down(x)

        # [n, (h / 4) * (w / 4), c / 2]
        theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)
        # [n, c / 2, (h / 8) * (w / 8)]
        phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)
        # [n, (h / 8) * (w / 8), c / 2]
        g = F.max_pool2d(self.g(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)
        # [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
        f = F.softmax(torch.bmm(theta, phi), 2)
        # [n, c / 2, h / 4, w / 4]
        y = torch.bmm(f, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))

        return x + F.upsample(self.z(y), size=x.size()[2:], mode='bilinear', align_corners=True)









# class DGNLB(nn.Module):
#     def __init__(self, in_channels):
#         super(DGNLB, self).__init__()
#
#         self.roll = nn.Conv2d(1, int(in_channels / 2), kernel_size=1)
#         self.ita = nn.Conv2d(1, int(in_channels / 2), kernel_size=1)
#
#         self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
#         self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
#         self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
#
#         self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
#         self.down.weight.data.fill_(1. / 16)
#
#         #self.down_depth = nn.Conv2d(1, 1, kernel_size=4, stride=4, groups=in_channels, bias=False)
#         #self.down_depth.weight.data.fill_(1. / 16)
#
#         self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)
#
#     def forward(self, x, depth):
#         n, c, h, w = x.size()
#         x_down = self.down(x)
#
#         depth_down = F.avg_pool2d(depth, kernel_size=(4,4))
#
#         # [n, (h / 4) * (w / 4), c / 2]
#         #roll = self.roll(depth_down).view(n, int(c / 2), -1).transpose(1, 2)
#         # [n, c / 2, (h / 4) * (w / 4)]
#         #ita = self.ita(depth_down).view(n, int(c / 2), -1)
#         # [n, (h / 4) * (w / 4), (h / 4) * (w / 4)]
#
#         depth_correlation = F.softmax(torch.bmm(depth_down.view(n, 1, -1).transpose(1, 2), depth_down.view(n, 1, -1)), 2)
#
#
#         # [n, (h / 4) * (w / 4), c / 2]
#         theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)
#         # [n, c / 2, (h / 8) * (w / 8)]
#         phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)
#         # [n, (h / 8) * (w / 8), c / 2]
#         g = F.max_pool2d(self.g(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)
#         # [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
#         f_correlation = F.softmax(torch.bmm(theta, phi), 2)
#         # [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
#         final_correlation = F.softmax(torch.bmm(depth_correlation, f_correlation), 2)
#
#         # [n, c / 2, h / 4, w / 4]
#         y = torch.bmm(final_correlation, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))
#
#         return x + F.upsample(self.z(y), size=x.size()[2:], mode='bilinear', align_corners=True)
