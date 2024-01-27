import kornia.losses
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from loss.ms_ssim import MSSSIM

class ReconstructionLoss_grp(nn.Module):
    # def __init__(self, alpha=1, beta=20, theta=5):
    def __init__(self):
        super(ReconstructionLoss_grp, self).__init__()

        self.l1_loss = nn.L1Loss()

        # self.alpha = alpha
        # self.beta = beta
        # self.theta = theta

    def forward(self, im_group1_extract, im_group2_extract, im_ir, im_vis):
        # ms_ssim_loss = (1 - self.ms_ssim(im_fus, im_ir)) + (1 - self.ms_ssim(im_fus, im_vi))
        # ms_ssim_loss_ir = (1 - self.ms_ssim(im_ir_extract, (map_ir * im_ir + map_vi * im_vi)))
        # ms_ssim_loss_vis = (1 - self.ms_ssim(im_vis_extract, (map_ir * im_ir + map_vi * im_vi)))
        l1_loss_group1 = self.l1_loss(im_group1_extract, im_ir)
        l1_loss_group2 = self.l1_loss(im_group2_extract, im_vis)

        l1_loss = l1_loss_group1 + l1_loss_group2

        loss = l1_loss
        # loss = self.alpha * ms_ssim_loss + self.beta * l1_loss + self.theta * grad_loss

        return loss