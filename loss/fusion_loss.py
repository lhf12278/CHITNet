import kornia.losses
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from loss.ms_ssim import MSSSIM
from loss.reconstruction_loss import ReconstructionLoss_grp
from loss.edge_loss import EdgeLoss

class FusionLoss_assist(nn.Module):
    def __init__(self):
        super(FusionLoss_assist, self).__init__()

        self.ms_ssim = MSSSIM()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.grad_loss = JointGrad()
        self.recon_loss = ReconstructionLoss_grp()
        self.edge_loss = EdgeLoss()
        self.kl = KLDivLoss()

        # self.alpha = alpha
        # self.beta = beta
        # self.theta = theta

        # def forward(self, im_fus, im_ir, im_vi):

    def forward(self, fuse_out_main, im_fus, im_ir, im_vi, map_ir, map_vi, out_group1, out_group2, feat_1_edge_img, feat_2_edge_img):
    # def forward(self, fuse_out_main, im_fus, im_ir, im_vi, map_ir, map_vi):
        loss_recon = self.recon_loss(out_group1, out_group2, im_ir, im_vi)
        loss_edge = self.edge_loss(feat_1_edge_img, feat_2_edge_img, im_ir, im_vi)
        l1_loss = self.l1_loss(im_fus, (map_ir * im_ir + map_vi * im_vi))
        # l1_loss = self.l1_loss(im_fus, im_ir) + self.l1_loss(im_fus, im_vi)
        grad_loss = self.grad_loss(im_fus, im_ir, im_vi)

        # inter_loss = self.kl(im_fus, fuse_out_main.detach())

        inter_loss = self.kl(im_fus, fuse_out_main)
        # inter_loss = self.l1_loss(im_fus, fuse_out_main)

        # loss = 0.5 * l1_loss + grad_loss
        # loss = 0.6 * l1_loss + grad_loss
        # loss = 0.3 * l1_loss + grad_loss
        # loss = grad_loss
        # loss = l1_loss + grad_loss
        loss = 0.05 * l1_loss + grad_loss + inter_loss + loss_recon + loss_edge    #ours

        # loss = l1_loss + grad_loss + inter_loss + loss_recon + loss_edge

    # loss = 0.05 * l1_loss + grad_loss + loss_recon + loss_edge
        return loss

class FusionLoss_main(nn.Module):
    # def __init__(self, alpha=1, beta=1, theta=0.5):
    #     super(FusionLoss_main, self).__init__()
    def __init__(self):
        super(FusionLoss_main, self).__init__()

        self.ms_ssim = MSSSIM()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.grad_loss = JointGrad()

        # self.alpha = alpha
        # self.beta = beta
        # self.theta = theta

    # def forward(self, im_fus, im_ir, im_vi):
    def forward(self, im_fus, im_ir, im_vi, map_ir, map_vi):
        
        l1_loss = self.l1_loss(im_fus, (map_ir * im_ir + map_vi * im_vi))
        # l1_loss = self.l1_loss(im_fus, im_ir) + self.l1_loss(im_fus, im_vi)
        grad_loss = self.grad_loss(im_fus, im_ir, im_vi)
        
        # loss = 0.5 * l1_loss + grad_loss
        # loss = 0.6 * l1_loss + grad_loss
        # loss = 0.3 * l1_loss + grad_loss
        # loss = grad_loss
        # loss = l1_loss + grad_loss
        loss = 0.05 * l1_loss + grad_loss    #ours
        # loss = l1_loss + grad_loss    #ours

        return loss

class JointGrad(nn.Module):
    def __init__(self):
        super(JointGrad, self).__init__()

        self.laplacian = kornia.filters.laplacian
        self.l1_loss = nn.L1Loss()

    def forward(self, im_fus, im_ir, im_vi):

        ir_grad = torch.abs(self.laplacian(im_ir, 3))
        vi_grad = torch.abs(self.laplacian(im_vi, 3))
        # fus_grad = torch.abs(self.laplacian(im_fus, 3))
        fus_grad = self.laplacian(im_fus, 3)

        JGrad = torch.where(ir_grad-vi_grad >= 0, self.laplacian(im_ir, 3), self.laplacian(im_vi, 3))
        loss_JGrad = self.l1_loss(JGrad, fus_grad)

        # loss_JGrad = self.l1_loss(torch.max(ir_grad, vi_grad), fus_grad)

        return loss_JGrad


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

    
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()

    def forward(self, p, q):

        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)

       # loss = F.kl_div(q.log(), p, reducion='batchmean')
        loss = F.kl_div(q.log(), p)

        return loss