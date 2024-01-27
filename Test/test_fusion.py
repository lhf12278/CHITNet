import sys

sys.path.append("..")

import argparse
import pathlib
import warnings
import statistics
import time
from thop import profile

import os
import cv2
import numpy as np
import kornia
import torch.backends.cudnn
import torch.cuda
import torch.utils.data
import torchvision
from torch import Tensor
from tqdm import tqdm

from dataloader.fuse_data_vsm import TestData
from models.AssistNet import AssistNet
from models.FusionNet import FusionNet1
from models.TransNet import transNet_norm_q

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def hyper_args():
    """
    get hyper parameters from args
    """
    parser = argparse.ArgumentParser(description='Fuse Net eval process')
    # dataset
    # parser.add_argument('--ir', default='../dataset/test/OTCBVS/ir', type=pathlib.Path)  # OTCBVS-orisize
    # parser.add_argument('--ir_reverse', default='../dataset/test/OTCBVS/ir_reverse', type=pathlib.Path)
    # parser.add_argument('--vis', default='../dataset/test/OTCBVS/vis', type=pathlib.Path)
    # parser.add_argument('--vis_reverse', default='../dataset/test/OTCBVS/vis_reverse', type=pathlib.Path)
    parser.add_argument('--ir', default='../dataset/test/RoadScene/ir', type=pathlib.Path)  # RoadScene-orisize
    parser.add_argument('--ir_reverse', default='../dataset/test/RoadScene/ir_reverse', type=pathlib.Path)
    parser.add_argument('--vis', default='../dataset/test/RoadScene/vis', type=pathlib.Path)
    parser.add_argument('--vis_reverse', default='../dataset/test/RoadScene/vis_reverse', type=pathlib.Path)
    # parser.add_argument('--ir', default='../dataset/test/TNO/ir', type=pathlib.Path) #TNO-orisize
    # parser.add_argument('--ir_reverse', default='../dataset/test/TNO/ir_reverse', type=pathlib.Path)
    # parser.add_argument('--vis', default='../dataset/test/TNO/vis', type=pathlib.Path)
    # parser.add_argument('--vis_reverse', default='../dataset/test/TNO/vis_reverse', type=pathlib.Path)

    # checkpoint
    parser.add_argument('--ckpt_ir',                                  #path for loss rate
                        default='../cache/irBranch/DataEnLRrand-5-gradR240124_single_stage0.05l1/fus_1000.pth',
                        help='checkpoint cache folder')
    parser.add_argument("--ckpt_vis",
                        default="../cache/visBranch/DataEnLRrand-5-gradR240124_single_stage0.05l1/fus_1000.pth",
                        type=str, help="path to pretrained model (default: none)")
    parser.add_argument('--ckpt_f0',
                        default='../cache/Fusion0/DataEnLRrand-5-gradR240124_single_stage0.05l1/fus_1000.pth',
                        help='checkpoint cache folder')
    parser.add_argument('--ckpt_f1',
                        default='../cache/Fusion1/DataEnLRrand-5-gradR240124_single_stage0.05l1/fus_1000.pth',
                        help='checkpoint cache folder')
    parser.add_argument('--ckpt_trans',
                        default='../cache/Trans/DataEnLRrand-5-gradR240124_single_stage0.05l1/fus_1000.pth',
                        help='checkpoint cache folder')
    parser.add_argument('--dst', default='../results/Roadscene/DataEnLRrand-5-gradR2401246_single_stage0.05l1-Roadscene1000/', # Roadscene-dataset
                        help='fuse image save folder', type=pathlib.Path)
    parser.add_argument('--dim', default=1, type=int, help='AFuse feather dim')
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")

    args = parser.parse_args()
    return args

def main(args):

    cuda = args.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    print("===> Loading datasets")
    data = TestData(args.ir, args.vis, args.ir_reverse, args.vis_reverse)
    # data = ExtractTestData_norever(args.ir, args.vis_warp)
    test_data_loader = torch.utils.data.DataLoader(data, 1, True, pin_memory=True)

    print("===> Building model")
    ir_net = AssistNet().to(device)
    vis_net = AssistNet().to(device)
    trans_net = transNet_norm_q().to(device)
    fuse_net = FusionNet1().to(device)

    print("===> loading trained assist model_ir '{}'".format(args.ckpt_ir))
    ir_model_state_dict = torch.load(args.ckpt_ir)
    ir_net.load_state_dict(ir_model_state_dict)

    print("===> loading trained assist model_vis '{}'".format(args.ckpt_vis))
    vis_model_state_dict = torch.load(args.ckpt_vis)
    vis_net.load_state_dict(vis_model_state_dict)

    print("===> loading trained transfer model '{}'".format(args.ckpt_trans))
    trans_model_state_dict = torch.load(args.ckpt_trans)
    trans_net.load_state_dict(trans_model_state_dict)

    print("===> loading trained fusion model '{}'".format(args.ckpt_f1))
    f_model_state_dict = torch.load(args.ckpt_f1)
    fuse_net.load_state_dict(f_model_state_dict)

    print("===> Starting Testing")
    random_channels_np = np.loadtxt('/home/l/data2/dky/Inject-main/Trainer/random_channels.txt', dtype=np.int32)
    print(random_channels_np)
    random_channels_np = torch.tensor(random_channels_np, dtype=torch.long)
    test(ir_net, vis_net, trans_net, random_channels_np, fuse_net, test_data_loader, args.dst, device)


def test(ir_net, vis_net, trans_net, random_channels_np, fuse_net, test_data_loader, dst, device):

    ir_net.eval()
    vis_net.eval()
    trans_net.eval()
    fuse_net.eval()

    fus_time = []
    tqdm_loader = tqdm(test_data_loader, disable=True)
    # for (ir, vis), (ir_path, vi_path) in tqdm_loader:
    for (ir, vis), (ir_path, vis_path), (ir_reverse, vis_reverse), _ in tqdm_loader:

        name, ext = os.path.splitext(os.path.basename(ir_path[0]))

        file_name = name + ext
        ir, vis = ir.cuda(), vis.cuda()
        ir_reverse, vis_reverse = ir_reverse.cuda(), vis_reverse.cuda()

        # Fusion
        torch.cuda.synchronize() if str(device) == 'cuda' else None
        start = time.time()
        with torch.no_grad():
            irb_dfeats, visb_dfeats, irb_dfeats_enhan, visb_dfeats_enhan = trans_net(ir, vis, ir_reverse, vis_reverse)
            irb_dfeats = irb_dfeats.permute(0, 2, 3, 1)
            visb_dfeats = visb_dfeats.permute(0, 2, 3, 1)
            irb_dfeats_enhan = irb_dfeats_enhan.permute(0, 2, 3, 1)
            visb_dfeats_enhan = visb_dfeats_enhan.permute(0, 2, 3, 1)
            # channel_map_ir_bef = torch.cat([irb_dfeats[:, :, :, i] for i in range(64)], dim=1)
            # channel_map_vis_bef = torch.cat([visb_dfeats[:, :, :, i] for i in range(64)], dim=1)
            # channel_map_ir_aft = torch.cat([irb_dfeats_enhan[:, :, :, i] for i in range(64)], dim=1)
            # channel_map_vis_aft = torch.cat([visb_dfeats_enhan[:, :, :, i] for i in range(64)], dim=1)
            channel_map_ir_bef = torch.cat([visb_dfeats[:, :, :, 52]], dim=1)
            channel_map_vis_bef = torch.cat([visb_dfeats[:, :, :, 52]], dim=1)
            channel_map_ir_aft = torch.cat([irb_dfeats_enhan[:, :, :, 52]], dim=1)
            channel_map_vis_aft = torch.cat([visb_dfeats_enhan[:, :, :, 52]], dim=1)
            irb_dfeats = irb_dfeats.permute(0, 3, 1, 2)
            visb_dfeats = visb_dfeats.permute(0, 3, 1, 2)
            irb_dfeats_enhan = irb_dfeats_enhan.permute(0, 3, 1, 2)
            visb_dfeats_enhan = visb_dfeats_enhan.permute(0, 3, 1, 2)
            out_group1_ir, out_group2_ir, irb_feat_1_edge_img, irb_feat_2_edge_img, fuse_out_ir, fuse_feats_ir, irb_feat_1_edge, irb_feat_2_edge = ir_net(irb_dfeats_enhan, random_channels_np)
            out_group1_vi, out_group2_vi, visb_feat_1_edge_img, visb_feat_2_edge_img, fuse_out_vis, fuse_feats_vis, visb_feat_1_edge, visb_feat_2_edge = vis_net(visb_dfeats_enhan, random_channels_np)
            fuse_out = fuse_net(irb_dfeats_enhan, visb_dfeats_enhan, fuse_feats_ir, irb_feat_1_edge, irb_feat_2_edge, fuse_feats_vis, visb_feat_1_edge, visb_feat_2_edge)
            # print(irb_dfeats_enhan.shape, visb_dfeats_enhan.shape, fuse_feats_ir.shape, irb_feat_1_edge.shape)
            # fuse_out = fuse_net(irb_dfeats_enhan, visb_dfeats_enhan, fuse_feats_ir, fuse_feats_vis) # no edge
            # fuse_out = fuse_net(irb_dfeats_enhan, visb_dfeats_enhan)

            lhc = out_group1_ir + out_group2_ir

        torch.cuda.synchronize() if str(device) == 'cuda' else None
        end = time.time()
        fus_time.append(end - start)

        # TODO: save fused images
        imsave(fuse_out, dst / 'fused' / file_name)
        imsave(channel_map_ir_bef, dst / 'channel_map_ir_bef' / file_name)
        imsave(channel_map_vis_bef, dst / 'channel_map_vis_bef' / file_name)
        imsave(channel_map_ir_aft, dst / 'channel_map_ir_aft' / file_name)
        imsave(channel_map_vis_aft, dst / 'channel_map_vis_aft' / file_name)
        imsave(fuse_out_ir, dst / 'fuse_out_ir' / file_name)
        imsave(fuse_out_vis, dst / 'fuse_out_vis' / file_name)
        imsave(ir, dst / 'ir' / file_name)
        imsave(vis, dst / 'vis' / file_name)

    # statistics time record
    fuse_mean = statistics.mean(fus_time[1:])
    print('fuse time (average): {:.4f}'.format(fuse_mean))
    print('fps (equivalence): {:.4f}'.format(1. / fuse_mean))
    inputrans1 = torch.randn(1, 1, 256, 256).cuda()
    inputrans2 = torch.randn(1, 1, 256, 256).cuda()
    inputrans3 = torch.randn(1, 1, 256, 256).cuda()
    inputrans4 = torch.randn(1, 1, 256, 256).cuda()
    inputir1 = torch.randn(1, 64, 256, 256).cuda()
    # inputir2 = torch.randn(1, 64, 256, 256).cuda()
    inputvis1 = torch.randn(1, 64, 256, 256).cuda()
    # inputvis2 = torch.randn(1, 64, 256, 256).cuda()
    input1 = torch.randn(1, 64, 256, 256).cuda()
    input2 = torch.randn(1, 64, 256, 256).cuda()
    input3 = torch.randn(1, 64, 256, 256).cuda()
    input4 = torch.randn(1, 64, 256, 256).cuda()
    input5 = torch.randn(1, 64, 256, 256).cuda()
    input6 = torch.randn(1, 64, 256, 256).cuda()
    input7 = torch.randn(1, 64, 256, 256).cuda()
    input8 = torch.randn(1, 64, 256, 256).cuda()
    flops_trans, params_trans = profile(trans_net, (inputrans1, inputrans2, inputrans3, inputrans4))
    flops_ir, params_ir = profile(ir_net, (inputir1, random_channels_np))
    flops_vis, params_vis = profile(vis_net, (inputvis1, random_channels_np))
    flops_fuse, params_fuse = profile(fuse_net, (input1, input2, input3, input4, input5, input6, input7, input8))
    print('flops_trans: %.4f G, params_trans:%.4f M' % (flops_trans / 1e9, params_trans / 1e6))
    print('flops_ir: %.4f G, params_ir:%.4f M' % (flops_ir / 1e9, params_ir / 1e6))
    print('flops_vis: %.4f G, params_vis:%.4f M' % (flops_vis / 1e9, params_vis / 1e6))
    print('flops_fuse: %.4f G, params_fuse:%.4f M' % (flops_fuse / 1e9, params_fuse / 1e6))
    print("Params(M_trans): %.4f" % (params_count(trans_net) / (1000 ** 2)))
    print("Params(M_ir): %.4f" % (params_count(ir_net) / (1000 ** 2)))
    print("Params(M_vis): %.4f" % (params_count(vis_net) / (1000 ** 2)))
    print("Params(M_fuse): %.4f" % (params_count(fuse_net) / (1000 ** 2)))


    pass
def imsave(im_s: [Tensor], dst: pathlib.Path, im_name: str = ''):
    """
    save images to path
    :param im_s: image(s)
    :param dst: if one image: path; if multiple images: folder path
    :param im_name: name of image
    """

    im_s = im_s if type(im_s) == list else [im_s]
    dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
    for im_ts, p in zip(im_s, dst):
        im_ts = im_ts.squeeze().cpu()

        # DKY
        im_ts = im_ts.numpy()
        im_ts = (im_ts - np.min(im_ts)) / (np.max(im_ts) - np.min(im_ts))
        im_ts = np.clip(im_ts * 255.0, 0., 255.)
        im_ts = torch.from_numpy(im_ts)

        p.parent.mkdir(parents=True, exist_ok=True)
        # im_cv = kornia.utils.tensor_to_image(im_ts) * 255.
        im_cv = kornia.utils.tensor_to_image(im_ts)
        cv2.imwrite(str(p), im_cv)


def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = hyper_args()
    main(args)