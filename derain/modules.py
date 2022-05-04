import os
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

class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the models
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out

class myTransformerBlock(nn.Module):
    """ Transformer block """
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the models
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x

class VIT(nn.Module):
    """  the full GPT language models, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model


        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))
        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)
        # regularization
        self.drop = nn.Dropout(embd_pdrop)
        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        self.con1_1 = nn.Conv2d(128,64,1,1,0)
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

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(x1)
        ir_fea = self.avgpool(x2)
        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)
        x = x.reshape(bs, 2*self.n_embd, self.vert_anchors, self.horz_anchors) # dim:(B, 2*C, H, W)
        # # 这样截取的方式, 是否采用映射的方式更加合理？
        # rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        # ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        # rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear',align_corners=True)
        # ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear',align_corners=True)
        return x1+F.interpolate(self.con1_1(x), size=x1.size()[2:], mode='bilinear', align_corners=True)


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
        self.res_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
        )

    def forward(self, x):
        res_conv = self.res_conv(x)
        return F.relu(x + res_conv)

