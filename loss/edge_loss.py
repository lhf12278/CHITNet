import kornia.losses
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from loss.ms_ssim import MSSSIM
    
class EdgeLoss(nn.Module):
    # def __init__(self, alpha=1, beta=20, theta=5):
    def __init__(self):
        super(EdgeLoss, self).__init__()

        # self.ms_ssim = MSSSIM()
        # self.l1_loss = nn.L1Loss()
        # self.l2_loss = nn.MSELoss()
        self.grad_loss = SingleGrad()

        # self.alpha = alpha
        # self.beta = beta
        # self.theta = theta

    def forward(self, feat_1_edge_img, feat_2_edge_img, im_ir, im_ir_reverse):
        # ms_ssim_loss = (1 - self.ms_ssim(im_fus, im_ir)) + (1 - self.ms_ssim(im_fus, im_vi))
        # ms_ssim_loss_ir = (1 - self.ms_ssim(orib_feat_1_edge_img, orib_feat_2_edge_imgim_ir_extract, (map_ir * im_ir + map_vi * im_vi)))
        # ms_ssim_loss_vis = (1 - self.ms_ssim(im_vis_extract, (map_ir * im_ir + map_vi * im_vi)))
        # l1_loss_group1 = self.l1_loss(im_group1_extract, im_ir) + self.l1_loss(im_group1_extract, im_vis)
        # l1_loss_group2 = self.l1_loss(im_group2_extract, im_ir) + self.l1_loss(im_group2_extract, im_vis)
        grad_loss_group1 = self.grad_loss(feat_1_edge_img, im_ir)
        grad_loss_group2 = self.grad_loss(feat_2_edge_img, im_ir_reverse)
        # ms_ssim_loss_group1 = (1 - self.ms_ssim(im_group1_extract, im_ir)) + (1 - self.ms_ssim(im_group1_extract, im_vis))
        # ms_ssim_loss_group2 = (1 - self.ms_ssim(im_group2_extract, im_ir)) + (1 - self.ms_ssim(im_group2_extract, im_vis))
        # fuse_loss = self.alpha * ms_ssim_loss + self.beta * l1_loss + self.theta * grad_loss
        # l1_loss = l1_loss_group1 + l1_loss_group2
        # ms_ssim_loss = ms_ssim_loss_group1 + ms_ssim_loss_group2
        grad_loss = grad_loss_group1 + grad_loss_group2
        loss = grad_loss
        # loss = self.alpha * ms_ssim_loss + self.beta * l1_loss + self.theta * grad_loss

        return loss


class SingleGrad(nn.Module):
    def __init__(self):
        super(SingleGrad, self).__init__()

        self.laplacian = kornia.filters.laplacian
        self.l1_loss = nn.L1Loss()

    def forward(self, im_fus, im_ir):

        ir_grad = self.laplacian(im_ir, 3)
        fus_grad = self.laplacian(im_fus, 3)

        loss_SGrad = self.l1_loss(ir_grad, fus_grad)

        return loss_SGrad
