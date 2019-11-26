from data import *
from utils.augmentations import SSDAugmentation, SSDAugmentation_two_stage_end2end
from layers.modules import MultiBoxLoss, MultiBoxLoss_offset, MultiBoxLoss_four_corners, MultiBoxLoss_four_corners_with_border
from ssd_two_stage_end2end import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from torchsummary import summary


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO', 'TWO_STAGE_END2END'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='voc_weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists('weights/' + args.save_folder):
    os.mkdir('weights/' + args.save_folder)


def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    elif args.dataset == 'TWO_STAGE_END2END':
        cfg = two_stage_end2end
        dataset = CAR_CARPLATE_TWO_STAGE_END2ENDDetection(root=args.dataset_root,
                                    transform=SSDAugmentation_two_stage_end2end(cfg['min_dim'],
                                                         MEANS),
                                    dataset_name='trainval')

    if args.visdom:
        import visdom
        global viz
        viz = visdom.Visdom()

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['min_dim_2'], cfg['num_classes'], cfg['expand_num'])
    net = ssd_net

    # summary
    # summary(net, input_size=(3, 300, 300))

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load('weights/' + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
        ssd_net.has_lp.apply(weights_init)
        ssd_net.size_lp.apply(weights_init)
        ssd_net.offset.apply(weights_init)

        ssd_net.vgg_2.apply(weights_init)
        ssd_net.loc_2.apply(weights_init)
        ssd_net.conf_2.apply(weights_init)
        ssd_net.four_corners_2.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999),
    #                        weight_decay=args.weight_decay)
    criterion = MultiBoxLoss_offset(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    # criterion_2 = MultiBoxLoss_four_corners(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
    #                          False, args.cuda)
    criterion_2 = MultiBoxLoss_four_corners_with_border(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                          False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    size_lp_loss = 0
    offset_loss = 0
    has_lp_loss = 0

    loc_2_loss = 0
    conf_2_loss = 0
    four_corners_2_loss = 0
    border_2_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Size LP Loss', 'Offset Loss', 'Has LP Loss',
                      'Loc2 loss', 'Conf2 Loss', 'Four Corners2 Loss', 'Border2 Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            epoch += 1
            update_vis_plot(epoch, loc_loss, conf_loss, size_lp_loss, offset_loss, has_lp_loss, loc_2_loss, conf_2_loss,
                            four_corners_2_loss, border_2_loss, epoch_plot, None, 'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            size_lp_loss = 0
            offset_loss = 0
            has_lp_loss = 0

            loc_2_loss = 0
            conf_2_loss = 0
            four_corners_2_loss = 0
            border_2_loss = 0

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images, targets)
        out_1 = out[:6]
        # backprop
        optimizer.zero_grad()
        # offset fine-tuning and pre-training(only loss_l and loss_c)
        loss_l, loss_c, loss_size_lp, loss_offset, loss_has_lp = criterion(out_1, targets)
        loss_1 = loss_l + loss_c + loss_size_lp + loss_offset + loss_has_lp
        # 第二阶段网络的损失,先看是否有target,target最后以为表示GT是否是valid,只有valid才能加入loss计算
        loc_2_data, conf_2_data, priors_2, four_corners_2_data = out[6:-1]
        targets_2 = out[-1]
        valid_idx = targets_2[:, -1] == 1
        targets_2 = targets_2[valid_idx, :-1]
        if targets_2.shape[0] > 0:
            loc_2_data = loc_2_data[valid_idx, :, :]
            conf_2_data = conf_2_data[valid_idx, :, :]
            four_corners_2_data = four_corners_2_data[valid_idx, :, :]
            out_2 = (loc_2_data,
                     conf_2_data,
                     priors_2,
                     four_corners_2_data)

            targets_2_list = []
            for i in range(targets_2.shape[0]):
                targets_2_list.append(targets_2[i, :].unsqueeze(0))
            loss_l_2, loss_c_2, loss_four_corners_2, loss_border_2 = criterion_2(out_2, targets_2_list)
            loss_2 = loss_l_2 + loss_c_2 + loss_four_corners_2 + loss_border_2
            loss = loss_1 + loss_2
        else:
            loss = loss_1

        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        size_lp_loss += loss_size_lp.data[0]
        offset_loss += loss_offset.data[0]
        has_lp_loss += loss_has_lp.data[0]
        if targets_2.shape[0] > 0:
            loc_2_loss += loss_l_2.data[0]
            conf_2_loss += loss_c_2.data[0]
            four_corners_2_loss += loss_four_corners_2.data[0]
            border_2_loss += loss_border_2.data[0]

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')

        if args.visdom:
            if targets_2.shape[0] > 0:
                update_vis_plot(iteration, loss_l.data[0], loss_c.data[0], loss_size_lp.data[0], loss_offset.data[0],
                                loss_has_lp.data[0], loss_l_2.data[0], loss_c_2.data[0],
                                loss_four_corners_2.data[0], loss_border_2.data[0], iter_plot, epoch_plot, 'append')
            else:
                update_vis_plot(iteration, loss_l.data[0], loss_c.data[0], loss_size_lp.data[0], loss_offset.data[0],
                                loss_has_lp.data[0], 0, 0, 0, 0, iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/' + args.save_folder + 'ssd300_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(),
               'weights/' + args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 10)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, size_lp, offset, has_lp, loc_2, conf_2, four_corners_2,
                    border_2, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 10)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, size_lp, offset, has_lp,
                        loc_2, conf_2, four_corners_2, border_2,
                        loc + conf + size_lp + offset + has_lp + loc_2 + conf_2 + four_corners_2 + border_2]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 10)).cpu(),
            Y=torch.Tensor([loc, conf, size_lp, offset, has_lp,
                            loc_2, conf_2, four_corners_2, border_2,
                            loc + conf + size_lp + offset + has_lp + loc_2 + conf_2 + four_corners_2 + border_2]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()