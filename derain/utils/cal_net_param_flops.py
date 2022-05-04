from thop import profile
from thop import clever_format
from torch import nn
import torch
import torch.nn.functional as F


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
        # self.cft = VIT(d_model=num_features)

        self.conv_DBR = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, dilation=1),
        )
        self.conv_DBR1 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=2, dilation=2),
        )
        self.conv_DBR2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, dilation=1),
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
        f = F.relu(f + self.conv_DBR(f))
        f = F.relu(f + self.conv_DBR1(f))
        f = F.relu(f + self.conv_DBR2(f))
        dps = self.conv_dps(dps)
        # f = self.cft(f, dps.detach())
        r = self.conv1(f)
        x = x + r
        x = (x * self.std + self.mean).clamp(min=0, max=1)
        return x



input = torch.randn(1, 3, 512, 1024)
dps = torch.randn(1,1,512,1024)
model = DDGN_Depth_CFT()
# stat(model,(3,512,1024))
flops, params = profile(model, inputs=(input.cpu(),dps.cpu()))
flops, params = clever_format([flops, params], "%.3f")
print('params:',params,'flops:',flops)

