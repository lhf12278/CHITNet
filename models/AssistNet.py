import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from einops import rearrange


class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate
    self.dense_layers = nn.Sequential(*modules)
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out

class FuseModule(nn.Module):
  """ Interactive fusion module"""
  def __init__(self, in_dim=64):
    super(FuseModule, self).__init__()
    self.chanel_in = in_dim

    self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
    self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

    self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
    self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
    self.sig = nn.Sigmoid()

  def forward(self, x, prior):
    x_q = self.query_conv(x)
    prior_k = self.key_conv(prior)
    energy = x_q * prior_k
    attention = self.sig(energy)
    attention_x = x * attention
    attention_p = prior * attention

    x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))
    x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

    p_gamma = self.gamma2(torch.cat((prior, attention_p), dim=1))
    prior_out = prior * p_gamma[:, [0], :, :] + attention_p * p_gamma[:, [1], :, :]

    return x_out, prior_out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Channel_Max_Pooling(torch.nn.MaxPool1d):
    def __init__(self, channels, isize):
        super(Channel_Max_Pooling, self).__init__(channels)
        self.kernel_size = channels
        self.stride = isize

    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = torch.nn.functional.max_pool1d(input, self.kernel_size, self.stride,
                                                self.padding, self.dilation, self.ceil_mode,
                                                self.return_indices)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        pooled = rearrange(pooled, 'b c (h w) -> b c h w', h=256, c=1)
        return pooled

class Channel_Avg_Pooling(torch.nn.MaxPool1d):
    def __init__(self, channels, isize):
        super(Channel_Avg_Pooling, self).__init__(channels)
        self.kernel_size = channels
        self.stride = isize

    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = torch.nn.functional.avg_pool1d(input, self.kernel_size, self.stride)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        pooled = rearrange(pooled, 'b c (h w) -> b c h w', h=256, c=1)
        return pooled

#通道
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        attn = self.softmax(out)
        out = x * attn + x
        return out

#空间
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        attn = self.sigmoid(x)
        out = x * attn + x
        return out


class CrossAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(CrossAttention, self).__init__()
        self.a1 = nn.Conv2d(32, 32, (1, 1))  # for feature(test
        self.b1 = nn.Conv2d(32, 32, (1, 1))
        self.c1 = nn.Conv2d(32, 32, (1, 1))
        self.out_conv1 = nn.Conv2d(32, 32, (1, 1))
        self.sm1 = nn.Softmax(dim=-1)

        # for ablation-no abundant-----------------------------
        # self.a1 = nn.Conv2d(64, 64, (1, 1))  # for feature(test
        # self.b1 = nn.Conv2d(64, 64, (1, 1))
        # self.c1 = nn.Conv2d(64, 64, (1, 1))
        # self.out_conv1 = nn.Conv2d(64, 64, (1, 1))
        # self.sm1 = nn.Softmax(dim=-1)  #

    def forward(self, q, v):
        A = self.a1(v)  # s2c1
        B = self.b1(v)

        C = self.c1(q)
        b, c, h, w = C.size()
        C = C.view(b, -1, w * h)  # C x HsWs
        b, c, h, w = B.size()
        B = B.view(b, -1, w * h).permute(0, 2, 1)  # HsWs x C

        S = torch.bmm(C, B)  # HcWc x HsWs
        S = self.sm1(S)  # style attention map

        b, c, h, w = A.size()
        A = A.view(b, -1, w * h)  # C x HsWs
        O = torch.bmm(S, A)  # C x HcWc

        O = O.view(b, c, h, w)
        O = self.out_conv1(O)
        O += v
        return O

class AssistNet(nn.Module):
  def __init__(self, nfeats=64):
    super(AssistNet, self).__init__()

    # head
    self.conv1_1 = nn.Sequential(
      nn.Conv2d(2, nfeats, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.2)
    )
    self.conv2_1 = nn.Sequential(
      nn.Conv2d(2, nfeats, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(negative_slope=0.2)
    )
    self.conv3_1 = nn.Sequential(
        nn.Conv2d(32, nfeats, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.2)
    )
    self.conv3_2 = nn.Sequential(
        nn.Conv2d(32, nfeats, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.2)
    )
    self.channel_shuffle = nn.ChannelShuffle(2)

    # body-densenet
    self.nChannels = nfeats
    self.nDenselayer = 3
    self.growthRate = nfeats
    Orib_path = []
    Reverb_path = []
    for i in range(1):
      Orib_path.append(RDB(self.nChannels, self.nDenselayer, self.growthRate))
      Reverb_path.append(RDB(self.nChannels, self.nDenselayer, self.growthRate))
    self.orib_path = nn.Sequential(*Orib_path)
    self.reverb_path = nn.Sequential(*Reverb_path)
    self.CA = CrossAttention()
    self.chanshu = nn.Conv2d(64, 64, (1, 1))

    # tail
    self.out_conv = nn.Conv2d(nfeats, 1, kernel_size=3, stride=1, padding=1)
    self.act = nn.Tanh()
    self.out_conv_group1 = nn.Conv2d(nfeats//2, 1, kernel_size=3, stride=1, padding=1)
    self.out_conv_group2 = nn.Conv2d(nfeats//2, 1, kernel_size=3, stride=1, padding=1)
    self.out_edge_group1 = nn.Conv2d(nfeats, 1, kernel_size=3, stride=1, padding=1)
    self.out_edge_group2 = nn.Conv2d(nfeats, 1, kernel_size=3, stride=1, padding=1)

  def forward(self, input_feats, random_channels):

      orib_dfeats = input_feats
      orib_feat_1 = orib_dfeats[:, random_channels, :, :]
      orib_feat_2 = orib_dfeats[:, 63 - random_channels, :, :]
      out_group1 = self.out_conv_group1(orib_feat_1)
      out_group1 = self.act(out_group1)
      out_group2 = self.out_conv_group2(orib_feat_2)
      out_group2 = self.act(out_group2)
      orib_feat_1_enhan = self.CA(orib_feat_2, orib_feat_1)
      orib_feat_1_edge = self.conv3_1(orib_feat_1_enhan)
      orib_feat_1_edge_img = self.out_edge_group1(orib_feat_1_edge)
      orib_feat_1_edge_img = self.act(orib_feat_1_edge_img)
      orib_feat_2_enhan = self.CA(orib_feat_1, orib_feat_2)
      orib_feat_2_edge = self.conv3_2(orib_feat_2_enhan)
      orib_feat_2_edge_img = self.out_edge_group1(orib_feat_2_edge)
      orib_feat_2_edge_img = self.act(orib_feat_2_edge_img)

      fu = torch.cat((orib_feat_1_enhan, orib_feat_2_enhan), dim=1)

      # tail
      out = self.out_conv(fu)
      out = self.act(out)

      return out_group1, out_group2, orib_feat_1_edge_img, orib_feat_2_edge_img, out, fu, orib_feat_1_edge, orib_feat_2_edge
      # return out_group, orib_feat_edge_img, out, fu, orib_feat_edge # for ablation-no abundant

def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()

if __name__ == '__main__':

    model = AssistNet(64).cuda()
    a = torch.randn(1, 1, 64, 64).cuda()
    b = model(a, a)
    print(b.shape)
    model = AssistNet(64).cuda()
    model.eval()
    print("Params(M): %.2f" % (params_count(model) / (1000 ** 2)))
    import time
    x = torch.Tensor(1, 1, 64, 64).cuda()

    N = 10
    with torch.no_grad():
        for _ in range(N):
            out = model(x, x)

        result = []
        for _ in range(N):
            torch.cuda.synchronize()
            st = time.time()
            for _ in range(N):
                out = model(x, x)
            torch.cuda.synchronize()
            result.append((time.time() - st)/N)
        print("Running Time: {:.3f}s\n".format(np.mean(result)))


