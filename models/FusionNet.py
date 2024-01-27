import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class FusionNet0(nn.Module):
  def __init__(self, nfeats=64):
    super(FusionNet0, self).__init__()

    self.fuse_res = nn.Conv2d(128, nfeats, kernel_size=3, stride=1, padding=1)

    # tail
    self.out_conv = nn.Conv2d(nfeats, 1, kernel_size=3, stride=1, padding=1)
    self.act = nn.Tanh()

  def forward(self, irb_dfeats_enhan, visb_dfeats_enhan):
  # def forward(self, ir, vi, ir_rever, vi_rever, fuse_feats_ori, orib_feat_edge, fuse_feats_rever, reverb_feat_edge):


    fu = torch.cat((irb_dfeats_enhan, visb_dfeats_enhan), dim=1)
    # fu = torch.cat((irb_dfeats_enhan, visb_dfeats_enhan, fuse_feats_ori, orib_feat_edge, fuse_feats_rever, reverb_feat_edge), dim=1)

    fuse_feats = self.fuse_res(fu)

    # tail
    out = self.out_conv(fuse_feats)
    out = self.act(out)

    # print('out', out.shape)

    return out


class FusionNet1(nn.Module):
  def __init__(self, nfeats=64):
    super(FusionNet1, self).__init__()

    # body-fuse
    # self.fuse = FuseModule()
    self.fuse_res = nn.Conv2d(512, nfeats, kernel_size=3, stride=1, padding=1)

    # tail
    self.out_conv = nn.Conv2d(nfeats, 1, kernel_size=3, stride=1, padding=1)
    self.act = nn.Tanh()

  def forward(self, irb_dfeats_enhan, visb_dfeats_enhan, fuse_feats_ori, orib_feat_1_edge, orib_feat_2_edge, fuse_feats_rever, reverb_feat_1_edge, reverb_feat_2_edge):
  # def forward(self, ir, vi, ir_rever, vi_rever, fuse_feats_ori, orib_feat_edge, fuse_feats_rever, reverb_feat_edge):

    fu = torch.cat((irb_dfeats_enhan, visb_dfeats_enhan, fuse_feats_ori, orib_feat_1_edge, orib_feat_2_edge, fuse_feats_rever, reverb_feat_1_edge, reverb_feat_2_edge), dim=1)
    # fu = torch.cat((irb_dfeats_enhan, visb_dfeats_enhan, fuse_feats_ori, orib_feat_edge, fuse_feats_rever, reverb_feat_edge), dim=1)

    fuse_feats = self.fuse_res(fu)

    # body-concat
    # fuse_feats = self.fuse_res(torch.cat((ir_dfeats, vi_dfeats), dim=1))

    # tail
    out = self.out_conv(fuse_feats)
    out = self.act(out)

    # print('out', out.shape)

    return out

def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()

