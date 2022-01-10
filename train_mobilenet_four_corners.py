from data import *
from utils.augmentations import SSDAugmentation_four_corners
from layers.modules import MultiBoxLoss_four_corners
from ssd_mobilenet_four_corners import build_ssd
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
from log import log
from evaluation import eval_results


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='CARPLATE_FOUR_CORNERS', choices=['CARPLATE_FOUR_CORNERS'],
                    type=str, help='CARPLATE_FOUR_CORNERS')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='mobilenet-v1-ssd-mp-0_675.pth',
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
parser.add_argument('--input_size', default=300, type=int, help='SSD300 or SSD512')
# ------------------------evaluation-------------------------------------------
parser.add_argument('--top_k', default=200, type=int,
                    help='Maximum number of predicted results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Minimum threshold of preserved results')
parser.add_argument('--eval_save_folder', default='eval/',
                    help='File path to save results')
parser.add_argument('--obj_type', default='carplate_four_corners', choices=['carplate_four_corners'],
                    type=str, help='carplate_four_corners')
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
    if args.dataset == 'CARPLATE_FOUR_CORNERS':
        cfg = mobilenet_carplate
        if args.input_size == 512:
            cfg = change_cfg_for_ssd512_mobilenet(cfg)
        dataset = CARPLATE_FOUR_CORNERSDetection(root=args.dataset_root,
                                    transform=SSDAugmentation_four_corners(cfg['min_dim'],
                                                         MEANS),
                                    dataset_name='trainval')
        from data import CARPLATE_FOUR_CORNERS_CLASSES as labelmap
        eval_dataset = CARPLATE_FOUR_CORNERSDetection(root=args.dataset_root,
                           transform=BaseTransform(args.input_size, MEANS),
                           target_transform=CARPLATE_FOUR_CORNERSAnnotationTransform(keep_difficult=True),
                           dataset_name='test')

    if args.visdom:
        import visdom
        global viz
        viz = visdom.Visdom()

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    # summary
    summary(net, input_size=(3, int(cfg['min_dim']), int(cfg['min_dim'])))

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        mobilenet_weights = torch.load('weights/' + args.basenet)
        base_net_weights = {".".join(k.split('.')[1:]):v for k, v in mobilenet_weights.items() if 'base_net' in k}
        extras_weights = {".".join(k.split('.')[1:]):v for k, v in mobilenet_weights.items() if 'extras' in k}
        print('Loading base network...')
        ssd_net.base_net.load_state_dict(base_net_weights)
        ssd_net.extras.load_state_dict(extras_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
        ssd_net.four_corners.apply(weights_init)

    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999),
                           weight_decay=args.weight_decay)
    criterion = MultiBoxLoss_four_corners(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    four_corners_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Four Corners Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    lr = args.lr
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            epoch += 1
            update_vis_plot(epoch, loc_loss, conf_loss, four_corners_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            four_corners_loss = 0

        if iteration in cfg['lr_steps']:
            step_index += 1
            lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            with torch.no_grad():
                targets = [Variable(ann.cuda()) for ann in targets]
        else:
            images = Variable(images)
            with torch.no_grad():
                targets = [Variable(ann) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_four_corners = criterion(out, targets)
        loss = loss_l + loss_c + loss_four_corners
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        four_corners_loss += loss_four_corners.item()

        if iteration % 100 == 0:
            log.l.info('''
                Timer: {:.5f} sec.\t LR: {}.\t Iter: {}.\t Loss_l: {:.5f}.\t Loss_c: {:.5f}.\t Loss_four_corners: {:.5f}.\t Loss: {:.5f}.
                '''.format((t1-t0), lr, iteration, loss_l.item(), loss_c.item(), loss_four_corners.item(),
                loss_l.item() + loss_c.item() + loss_four_corners.item()))

        if args.visdom:
            update_vis_plot(iteration, loss_l.item(), loss_c.item(), loss_four_corners.item(),
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/' + args.save_folder + 'mobilenet_ssd' + 
            str(args.input_size) + '_' + repr(iteration) + '.pth')

            # load net for evaluation
            num_classes = len(labelmap) + 1  # +1 for background
            eval_net = build_ssd('test', args.input_size, num_classes)  # initialize SSD
            eval_net.load_state_dict(torch.load('weights/' + args.save_folder + 'mobilenet_ssd' + str(args.input_size) + '_' + repr(iteration) + '.pth'))
            eval_net.eval()
            print('Finished loading model!')
            if args.cuda:
                eval_net = eval_net.cuda()
                cudnn.benchmark = True
            # evaluation begin
            eval_results.test_net(args.eval_save_folder, args.obj_type, args.dataset_root, 'test',
                    labelmap, eval_net, args.cuda, eval_dataset, BaseTransform(eval_net.size, MEANS), args.top_k,
                    args.input_size, thresh=args.confidence_threshold)

    torch.save(ssd_net.state_dict(),
               'weights/' + args.save_folder + '' + args.dataset + str(args.input_size) + '_mobilenet' + '.pth')
    # load net for evaluation for the final model
    num_classes = len(labelmap) + 1  # +1 for background
    eval_net = build_ssd('test', args.input_size, num_classes)  # initialize SSD
    eval_net.load_state_dict(torch.load('weights/' + args.save_folder + '' + args.dataset + str(args.input_size) + '_mobilenet' + '.pth'))
    eval_net.eval()
    print('Finished loading model!')
    if args.cuda:
        eval_net = eval_net.cuda()
        cudnn.benchmark = True
    # evaluation begin
    eval_results.test_net(args.eval_save_folder, args.obj_type, args.dataset_root, 'test',
            labelmap, eval_net, args.cuda, eval_dataset, BaseTransform(eval_net.size, MEANS), args.top_k,
            args.input_size, thresh=args.confidence_threshold)


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * 5) 
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 4)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, four_corners, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 4)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, four_corners, loc + conf + four_corners]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 4)).cpu(),
            Y=torch.Tensor([loc, conf, four_corners, loc + conf + four_corners]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
