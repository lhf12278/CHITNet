import sys

sys.path.append("..")

# import visdom
import cv2
from torch import Tensor
import kornia

import pathlib
import warnings
import logging.config
import argparse, os
import numpy as np

import numpy
import torch.backends.cudnn
import torch.utils.data
import torch.nn.functional

from tqdm import tqdm
from models.AssistNet import AssistNet
from dataloader.fuse_data_vsm import TrainingData
from models.TransNet import transNet_norm_q
from models.FusionNet import FusionNet0
from models.FusionNet import FusionNet1
from loss.fusion_loss import FusionLoss_assist
from loss.fusion_loss import FusionLoss_main

import setproctitle
setproctitle.setproctitle('dky_fusion_final')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def hyper_args():
    """
    get hyper parameters from args
    """
    parser = argparse.ArgumentParser(description='RobF Net train process')

    # dataset
    parser.add_argument('--ir', default='../dataset/train/RoadScene/ir', type=pathlib.Path)  # data enhancement-crop
    parser.add_argument('--ir_reverse', default='../dataset/train/RoadScene/ir_reverse', type=pathlib.Path)
    parser.add_argument('--vis', default='../dataset/train/RoadScene/vis', type=pathlib.Path)
    parser.add_argument('--vis_reverse', default='../dataset/train/RoadScene/vis_reverse', type=pathlib.Path)
    parser.add_argument('--ir_map', default='../dataset/train/svs_map/ir_map', type=pathlib.Path)
    parser.add_argument('--vis_map', default='../dataset/train/svs_map/vi_map', type=pathlib.Path)
    # parser.add_argument('--ir_reverse_map', default='../dataset/train/svs_map/ir_map', type=pathlib.Path)
    # parser.add_argument('--vis_reverse_map', default='../dataset/train/svs_map/vi_map', type=pathlib.Path)
    # train loss weights
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=0.01, type=float) # b1, th0.5
    parser.add_argument('--theta', default=1.0, type=float)
    # implement details
    # parser.add_argument('--dim', default=64, type=int, help='AFuse feather dim')8
    parser.add_argument('--dim', default=1, type=int, help='AFuse feather dim')
    parser.add_argument('--batchsize', default=8, type=int, help='mini-batch size')  # 32
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_fus1', default=0.00001, type=float, help='learning rate')
    parser.add_argument('--lr_assist', default=0.001, type=float, help='learning rate')
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument('--nEpochs', default=1000, type=int, help='number of total epochs to run')
    # parser.add_argument('--nEpochs_main', default=600, type=int, help='number of total main branch epochs to run')
    # parser.add_argument('--nEpochs_assist', default=400, type=int, help='number of total assist branch epochs to run')
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")
    # parser.add_argument("--step", type=int, default=1000, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument('--resume', default='', help='resume checkpoint')
    # parser.add_argument('--interval', default=20, help='record interval')
    parser.add_argument('--interval', default=200, help='record interval')
    # checkpoint
    parser.add_argument("--load_model_fuse", default=None, help="path to pretrained model (default: none)")
    parser.add_argument("--pretrained_ir", default=None, type=str, help="path to pretrained model (default: none)")
    parser.add_argument('--ckpt_ir', default='../cache/irBranch/DataEnLRrand-5-gradR240124_single_stage0.05l1', help='checkpoint cache folder')
    parser.add_argument("--pretrained_vis", default=None, type=str, help="path to pretrained model (default: none)")
    parser.add_argument("--ckpt_vis", default="../cache/visBranch/DataEnLRrand-5-gradR240124_single_stage0.05l1", type=str, help="path to pretrained model (default: none)")
    parser.add_argument('--ckpt_f0', default='../cache/Fusion0/DataEnLRrand-5-gradR240124_single_stage0.05l1', help='checkpoint cache folder')
    parser.add_argument('--ckpt_f1', default='../cache/Fusion1/DataEnLRrand-5-gradR240124_single_stage0.05l1', help='checkpoint cache folder')
    parser.add_argument('--ckpt_trans', default='../cache/Trans/DataEnLRrand-5-gradR240124_single_stage0.05l1', help='checkpoint cache folder')

    args = parser.parse_args()
    return args

# def main(args, visdom):
def main(args):

    cuda = args.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    log = logging.getLogger()

    epoch = args.nEpochs
    interval = args.interval

    print("===> Creating Save Path of Checkpoints")
    cache_f0 = pathlib.Path(args.ckpt_f0)
    cache_f1 = pathlib.Path(args.ckpt_f1)
    cache_trans = pathlib.Path(args.ckpt_trans)
    cache_ir = pathlib.Path(args.ckpt_ir)
    cache_vis = pathlib.Path(args.ckpt_vis)

    print("===> Loading datasets")
    # data = TrainingData(args.ir, args.vis, args.ir_reverse, args.vis_reverse, args.ir_map, args.vis_map, args.ir_reverse_map, args.vis_reverse_map)
    data = TrainingData(args.ir, args.vis, args.ir_reverse, args.vis_reverse, args.ir_map, args.vis_map)
    training_data_loader = torch.utils.data.DataLoader(data, args.batchsize, True, pin_memory=True)

    print("===> Building models")
    ir_net = AssistNet().to(device)
    vis_net = AssistNet().to(device)
    TransNet = transNet_norm_q().to(device)
    FuseNet0 = FusionNet0().to(device)
    FuseNet1 = FusionNet1().to(device)

    print("===> Defining Loss fuctions")
    criterion_ir = FusionLoss_assist().to(device)
    criterion_vis = FusionLoss_assist().to(device)
    criterion_fus = FusionLoss_main().to(device)

    print("===> Setting Optimizers")
    optimizer_ir = torch.optim.Adam(params=ir_net.parameters(), lr=args.lr)
    optimizer_vis = torch.optim.Adam(params=vis_net.parameters(), lr=args.lr)
    optimizer_fus0 = torch.optim.Adam(params=FuseNet0.parameters(), lr=args.lr)
    optimizer_fus1 = torch.optim.Adam(params=FuseNet1.parameters(), lr=args.lr)

    # TODO: optionally copy weights from a checkpoint
    if args.pretrained_ir is not None:
        print('Loading pre-trained AssistNet_ir checkpoint %s' % args.pretrained_ir)
        log.info(f'Loading pre-trained checkpoint {str(args.pretrained_ir)}')
        state = torch.load(str(args.pretrained_ir))
        ir_net.load_state_dict(state)
    else:
        print("=> no model found at '{}'".format(args.pretrained_ir))

    if args.pretrained_vis is not None:
        print('Loading pre-trained AssistNet_vis checkpoint %s' % args.pretrained_vis)
        log.info(f'Loading pre-trained checkpoint {str(args.pretrained_vis)}')
        state = torch.load(str(args.pretrained_vis))
        vis_net.load_state_dict(state)
    else:
        print("=> no model found at '{}'".format(args.pretrained_vis))

    if args.load_model_fuse is not None:
        print('Loading pre-trained FuseNet checkpoint %s' % args.load_model_fuse)
        log.info(f'Loading pre-trained checkpoint {str(args.load_model_fuse)}')
        state = torch.load(str(args.load_model_fuse))
        FuseNet0.load_state_dict(state)
    else:
        print("=> no model found at '{}'".format(args.load_model_fuse))

    print("===> Starting Training")

    # for original training
    # num_channels = 32
    # random_channels = torch.randperm(64)[:num_channels]
    # random_channels_np = random_channels.numpy()
    # # random_channels_np = numpy.array(random_channels_np)
    # print(random_channels_np)
    # # np.savetxt('random_channels.txt', random_channels_np, newline='')
    # # np.savetxt('random_channels.txt', random_channels_np, fmt='%d', newline='')
    # np.savetxt('random_channels.txt', random_channels_np, fmt='%d')

    # for ablation
    random_channels_np = np.loadtxt('/home/l/data2/dky/Inject-main/Trainer/random_channels.txt', dtype=np.int32)
    print(random_channels_np)
    random_channels = torch.tensor(random_channels_np, dtype=torch.long)

    for epoch in range(args.start_epoch, args.nEpochs + 1):
        tqdm_loader = tqdm(training_data_loader, disable=True)
        if epoch // interval == 0:
            train_main0(args, tqdm_loader, optimizer_fus0, TransNet, FuseNet0, criterion_fus, epoch)
        elif epoch // interval == 1:
            train_assist0(args, tqdm_loader, optimizer_ir, optimizer_vis, ir_net, vis_net, TransNet, random_channels, FuseNet0, criterion_ir, criterion_vis, epoch)
        elif epoch // interval == 3:
            train_assist1(args, tqdm_loader, optimizer_ir, optimizer_vis, ir_net, vis_net, TransNet, random_channels, FuseNet1, criterion_ir, criterion_vis, epoch)
        else:
            train_main1(args, tqdm_loader, optimizer_fus1, ir_net, vis_net, TransNet, random_channels, FuseNet1, criterion_fus, epoch)

        # TODO: save checkpoint
        save_checkpoint(TransNet, epoch, cache_trans) if epoch % 20 == 0 else None
        save_checkpoint(FuseNet0, epoch, cache_f0) if epoch % 20 == 0 else None
        save_checkpoint(FuseNet1, epoch, cache_f1) if epoch % 20 == 0 else None
        save_checkpoint(ir_net, epoch, cache_ir) if epoch % 20 == 0 else None
        save_checkpoint(vis_net, epoch, cache_vis) if epoch % 20 == 0 else None


def train_main0(args, tqdm_loader, optimizer_main, TransNet, FuseNet0, criterion_fus, epoch):

    TransNet.train()
    FuseNet0.train()
    # TODO: update learning rate of the optimizer
    lr_F = adjust_learning_rate(args, optimizer_main, epoch - 1)
    print("Epoch_main={}, lr_F={} ".format(epoch, lr_F))

    loss_total, loss_reg, loss_fus = [], [], []
    for (ir, vis), (ir_path, vis_path), (ir_reverse, vis_reverse), (ir_map, vi_map) in tqdm_loader: #for both ori & reverse loss

        name, ext = os.path.splitext(os.path.basename(ir_path[0]))
        file_name = name + ext

        ir, vis = ir.cuda(), vis.cuda()
        ir_reverse, vis_reverse = ir_reverse.cuda(), vis_reverse.cuda()
        ir_map, vi_map = ir_map.cuda(), vi_map.cuda()

        irb_dfeats, visb_dfeats, irb_dfeats_enhan, visb_dfeats_enhan = TransNet(ir, vis, ir_reverse, vis_reverse)
        fuse_out = FuseNet0(irb_dfeats_enhan, visb_dfeats_enhan)

        loss = criterion_fus(fuse_out, ir, vis, ir_map, vi_map)

        optimizer_main.zero_grad()
        loss.backward()
        optimizer_main.step()

        # if tqdm_loader.n % 40 == 0:
        #     show = torch.stack([ir_reg[0], vi[0], fuse_out[0]])
        #     visdom.images(show, win='Fusion')

        loss_total.append(loss.item())
    loss_avg = numpy.mean(loss_total)
    print('loss_avg', loss_avg)

    # TODO: visdom display
    # visdom.line([loss_avg], [epoch], win='loss-Fusion', name='total', opts=dict(title='Total-loss'), update='append' if epoch else '')


def train_main1(args, tqdm_loader, optimizer_fus, ir_net, vis_net, TransNet, random_channels, FuseNet1, criterion_fus, epoch):

    ir_net.eval()
    vis_net.eval()
    TransNet.train()
    FuseNet1.train()
    # TODO: update learning rate of the optimizer
    lr_F = adjust_learning_rate_1(args, optimizer_fus, epoch - 1)
    print("Epoch_main={}, lr_F={} ".format(epoch, lr_F))

    loss_total, loss_reg, loss_fus = [], [], []
    for (ir, vis), _, (ir_reverse, vis_reverse), (ir_map, vi_map) in tqdm_loader: #for both ori & reverse loss

        ir, vis = ir.cuda(), vis.cuda()
        ir_reverse, vis_reverse = ir_reverse.cuda(), vis_reverse.cuda()
        ir_map, vi_map = ir_map.cuda(), vi_map.cuda()

        irb_dfeats, visb_dfeats, irb_dfeats_enhan, visb_dfeats_enhan = TransNet(ir, vis, ir_reverse, vis_reverse)

        with torch.no_grad():

            out_group1, out_group2, irb_feat_1_edge_img, irb_feat_2_edge_img, fuse_out_ir, fuse_feats_ir, irb_feat_1_edge, irb_feat_2_edge = ir_net(irb_dfeats_enhan, random_channels)
            out_group1, out_group2, visb_feat_1_edge_img, visb_feat_2_edge_img, fuse_out_vis, fuse_feats_vis, visb_feat_1_edge, visb_feat_2_edge = vis_net(visb_dfeats_enhan, random_channels)

        fuse_out = FuseNet1(irb_dfeats_enhan, visb_dfeats_enhan, fuse_feats_ir, irb_feat_1_edge, irb_feat_2_edge, fuse_feats_vis, visb_feat_1_edge, visb_feat_2_edge)

        loss = criterion_fus(fuse_out, ir, vis, ir_map, vi_map)

        optimizer_fus.zero_grad()
        loss.backward()
        optimizer_fus.step()

        # if tqdm_loader.n % 40 == 0:
        #     show = torch.stack([ir_reg[0], vi[0], fuse_out[0]])
        #     visdom.images(show, win='Fusion')

        loss_total.append(loss.item())
    loss_avg = numpy.mean(loss_total)
    print('loss_avg', loss_avg)
    # TODO: visdom display
    # visdom.line([loss_avg], [epoch], win='loss-Fusion', name='total', opts=dict(title='Total-loss'), update='append' if epoch else '')


def train_assist0(args, tqdm_loader, optimizer_ir, optimizer_vis, ir_net, vis_net, TransNet, random_channels, FuseNet0, criterion_ir, criterion_vis, epoch):

    ir_net.train()
    vis_net.train()
    TransNet.eval()
    FuseNet0.eval()
    # TODO: update learning rate of the optimizer
    lr_I = adjust_learning_rate_as(args, optimizer_ir, epoch - 1)
    lr_V = adjust_learning_rate_as(args, optimizer_vis, epoch - 1)
    print("Epoch_assist={}, lr_I={} ".format(epoch, lr_I))
    print("Epoch_assist={}, lr_V={} ".format(epoch, lr_V))

    loss_total_ir, loss_total_vis, loss_fus = [], [], []
    for (ir, vis), _, (ir_reverse, vis_reverse), (ir_map, vi_map) in tqdm_loader: #for both ori & reverse loss

        ir, vis = ir.cuda(), vis.cuda()
        ir_reverse, vis_reverse = ir_reverse.cuda(), vis_reverse.cuda()
        ir_map, vi_map = ir_map.cuda(), vi_map.cuda()

        with torch.no_grad():
            irb_dfeats, visb_dfeats, irb_dfeats_enhan, visb_dfeats_enhan = TransNet(ir, vis, ir_reverse, vis_reverse)
            fuse_out = FuseNet0(irb_dfeats_enhan, visb_dfeats_enhan)
        out_group1_ir, out_group2_ir, irb_feat_1_edge_img, irb_feat_2_edge_img, fuse_out_ir, fuse_feats_ir, irb_feat_1_edge, irb_feat_2_edge = ir_net(irb_dfeats_enhan, random_channels)
        out_group1_vis, out_group2_vis, visb_feat_1_edge_img, visb_feat_2_edge_img, fuse_out_vis, fuse_feats_vis, visb_feat_1_edge, visb_feat_2_edge = vis_net(visb_dfeats_enhan, random_channels)

        loss_ir = criterion_ir(fuse_out, fuse_out_ir, ir, vis, ir_map, vi_map, out_group1_ir, out_group2_ir, irb_feat_1_edge_img, irb_feat_2_edge_img)
        loss_vis = criterion_vis(fuse_out, fuse_out_vis, ir, vis, ir_map, vi_map, out_group1_vis, out_group2_vis, visb_feat_1_edge_img, visb_feat_2_edge_img)

        optimizer_ir.zero_grad()
        loss_ir.backward()
        optimizer_ir.step()

        optimizer_vis.zero_grad()
        loss_vis.backward()
        optimizer_vis.step()

        # if tqdm_loader.n % 40 == 0:
        #     show = torch.stack([ir_reg[0], vi[0], fuse_out[0]])
        #     visdom.images(show, win='Fusion')

        loss_total_ir.append(loss_ir.item())
        loss_total_vis.append(loss_vis.item())
    loss_avg_ir = numpy.mean(loss_total_ir)
    loss_avg_vis = numpy.mean(loss_total_vis)
    print('loss_avg_ir', loss_avg_ir)
    print('loss_avg_vis', loss_avg_vis)
    # TODO: visdom display
    # visdom.line([loss_avg], [epoch], win='loss-Fusion', name='total', opts=dict(title='Total-loss'), update='append' if epoch else '')


def train_assist1(args, tqdm_loader, optimizer_ir, optimizer_vis, ir_net, vis_net, TransNet, random_channels, FuseNet1, criterion_ir, criterion_vis, epoch):

    ir_net.train()
    vis_net.train()
    TransNet.eval()
    FuseNet1.eval()
    # TODO: update learning rate of the optimizer
    lr_I = adjust_learning_rate_as(args, optimizer_ir, epoch - 1)
    lr_V = adjust_learning_rate_as(args, optimizer_vis, epoch - 1)
    print("Epoch_assist={}, lr_I={} ".format(epoch, lr_I))
    print("Epoch_assist={}, lr_V={} ".format(epoch, lr_V))

    loss_total_ir, loss_total_vis, loss_fus = [], [], []
    for (ir, vis), _, (ir_reverse, vis_reverse), (ir_map, vi_map) in tqdm_loader: #for both ori & reverse loss

        ir, vis = ir.cuda(), vis.cuda()
        ir_reverse, vis_reverse = ir_reverse.cuda(), vis_reverse.cuda()
        ir_map, vi_map = ir_map.cuda(), vi_map.cuda()

        with torch.no_grad():
            irb_dfeats, visb_dfeats, irb_dfeats_enhan, visb_dfeats_enhan = TransNet(ir, vis, ir_reverse, vis_reverse)

        out_group1_ir, out_group2_ir, irb_feat_1_edge_img, irb_feat_2_edge_img, fuse_out_ir, fuse_feats_ir, irb_feat_1_edge, irb_feat_2_edge = ir_net(irb_dfeats_enhan, random_channels)
        out_group1_vis, out_group2_vis, visb_feat_1_edge_img, visb_feat_2_edge_img, fuse_out_vis, fuse_feats_vis, visb_feat_1_edge, visb_feat_2_edge = vis_net(visb_dfeats_enhan, random_channels)
        with torch.no_grad():
            fuse_out = FuseNet1(irb_dfeats_enhan, visb_dfeats_enhan, fuse_feats_ir, irb_feat_1_edge, irb_feat_2_edge, fuse_feats_vis, visb_feat_1_edge, visb_feat_2_edge)

        loss_ir = criterion_ir(fuse_out, fuse_out_ir, ir, vis, ir_map, vi_map, out_group1_ir, out_group2_ir, irb_feat_1_edge_img, irb_feat_2_edge_img)
        loss_vis = criterion_vis(fuse_out, fuse_out_vis, ir, vis, ir_map, vi_map, out_group1_vis, out_group2_vis, visb_feat_1_edge_img, visb_feat_2_edge_img)

        optimizer_ir.zero_grad()
        loss_ir.backward()
        optimizer_ir.step()

        optimizer_vis.zero_grad()
        loss_vis.backward()
        optimizer_vis.step()

        # if tqdm_loader.n % 40 == 0:
        #     show = torch.stack([ir_reg[0], vi[0], fuse_out[0]])
        #     visdom.images(show, win='Fusion')

        loss_total_ir.append(loss_ir.item())
        loss_total_vis.append(loss_vis.item())
    loss_avg_ir = numpy.mean(loss_total_ir)
    loss_avg_vis = numpy.mean(loss_total_vis)
    print('loss_avg_ir', loss_avg_ir)
    print('loss_avg_vis', loss_avg_vis)
    # TODO: visdom display
    # visdom.line([loss_avg], [epoch], win='loss-Fusion', name='total', opts=dict(title='Total-loss'), update='append' if epoch else '')


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def adjust_learning_rate_1(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    real_epoch = epoch // 400
    lr = args.lr_fus1 * (0.1 ** (real_epoch - 1)) * (0.1 ** (real_epoch - 1)) * (0.1 ** ((epoch - 400 * real_epoch) // args.step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def adjust_learning_rate_as(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    real_epoch = epoch // 200
    lr = args.lr_assist * (0.1 ** (real_epoch - 1)) * (0.1 ** ((epoch - 200 * real_epoch) // args.step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def save_checkpoint(net, epoch, cache):
    model_folder = cache
    model_out_path = str(model_folder / f'fus_{epoch:04d}.pth')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(net.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = hyper_args()
    # visdom = visdom.Visdom(port=8097, env='Fusion')

    main(args)