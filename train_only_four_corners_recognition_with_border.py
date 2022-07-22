from data import *
from utils.augmentations import SSDAugmentation_four_corners
from layers.modules import MultiBoxLoss_only_four_corners_with_CIoU
from ssd_only_four_corners_recognition import build_ssd
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


def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='CARPLATE_ONLY_FOUR_CORNERS_WITH_BORDER_RECOGNITION', choices=['CARPLATE_ONLY_FOUR_CORNERS_WITH_BORDER_RECOGNITION'],
                    type=str, help='CARPLATE_ONLY_FOUR_CORNERS_WITH_BORDER_RECOGNITION')
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
parser.add_argument('--input_size', default=300, type=int, help='SSD300 or SSD512')
# ------------------------evaluation-------------------------------------------
parser.add_argument('--top_k', default=200, type=int,
                    help='Maximum number of predicted results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Minimum threshold of preserved results')
parser.add_argument('--eval_save_folder', default='eval/',
                    help='File path to save results')
parser.add_argument('--obj_type', default='carplate_only_four_corners_with_border_recognition', choices=['carplate_only_four_corners_with_border_recognition'],
                    type=str, help='carplate_only_four_corners_with_border_recognition')
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
    T_length = 18 # args.lpr_max_len

    if args.dataset == 'CARPLATE_ONLY_FOUR_CORNERS_WITH_BORDER_RECOGNITION':
        cfg = carplate
        if args.input_size == 512:
            cfg = change_cfg_for_ssd512(cfg)
        dataset = CARPLATE_FOUR_CORNERS_RECOGNITIONDetection(root=args.dataset_root,
                                    transform=SSDAugmentation_four_corners(cfg['min_dim'],
                                                         MEANS),
                                    dataset_name='trainval')
        from data import CARPLATE_FOUR_CORNERS_RECOGNITION_CLASSES as labelmap
        eval_dataset = CARPLATE_FOUR_CORNERS_RECOGNITIONDetection(root=args.dataset_root,
                           transform=BaseTransform(args.input_size, MEANS),
                           target_transform=CARPLATE_FOUR_CORNERS_RECOGNITIONAnnotationTransform(keep_difficult=True),
                           dataset_name='test')

    if args.visdom:
        import visdom
        global viz
        viz = visdom.Visdom()

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    # summary
    # summary(net, input_size=(3, int(cfg['min_dim']), int(cfg['min_dim'])))

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
        ssd_net.conf.apply(weights_init)
        ssd_net.four_corners.apply(weights_init)

    backbone_params = list(map(id, ssd_net.backbone.parameters()))
    container_params = list(map(id, ssd_net.container.parameters()))
    base_params = filter(lambda p: id(p) not in backbone_params+container_params, ssd_net.parameters())
    params = [{'params': base_params},
              {'params': ssd_net.backbone.parameters(), 'lr': args.lr*50.0},
              {'params': ssd_net.container.parameters(), 'lr': args.lr*50.0}]
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    optimizer = optim.Adam(params, lr=args.lr, betas=(0.9, 0.999),
                           weight_decay=args.weight_decay)
    criterion = MultiBoxLoss_only_four_corners_with_CIoU(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    rec_criterion = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean')

    net.train()
    # loss counters
    conf_loss = 0
    four_corners_loss = 0
    border_loss = 0
    rec_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Conf Loss', 'Four Corners Loss', 'Border Loss', 'Rec Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_recognition_collate,
                                  pin_memory=True)

    lr = args.lr
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            epoch += 1
            update_vis_plot(epoch, conf_loss, four_corners_loss, border_loss, rec_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            conf_loss = 0
            four_corners_loss = 0
            border_loss = 0
            rec_loss = 0

        if iteration in cfg['lr_steps']:
            step_index += 1
            lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        try:
            images, targets, labels, lengths = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets, labels, lengths = next(batch_iterator)

        # get ctc parameters
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)

        if args.cuda:
            images = Variable(images, requires_grad=False).cuda()
            with torch.no_grad():
                targets = Variable(targets.cuda())
            labels = Variable(labels, requires_grad=False).cuda()
        else:
            images = Variable(images, requires_grad=False)
            with torch.no_grad():
                targets = Variable(targets)
            labels = Variable(labels, requires_grad=False)

        # forward
        t0 = time.time()
        det_out, rec_out = net(images, targets)
        log_probs = rec_out.permute(2, 0, 1) # for ctc loss: T x N x C
        log_probs = log_probs.log_softmax(2).requires_grad_()
        # backprop
        optimizer.zero_grad()
        loss_c, loss_four_corners, loss_border = criterion(det_out, targets)
        loss_rec = rec_criterion(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        loss = loss_c + loss_four_corners + loss_border + loss_rec
        loss.backward()
        optimizer.step()
        t1 = time.time()
        conf_loss += loss_c.item()
        four_corners_loss += loss_four_corners.item()
        border_loss += loss_border.item()
        rec_loss += loss_rec.item()

        if iteration % 100 == 0:
            log.l.info('''
                Timer: {:.5f} sec.\t LR: {}.\t Iter: {}.\t Loss_c: {:.5f}.\t Loss_four_corners: {:.5f}.\t Loss_border: {:.5f}.\t Loss_rec: {:.5f}.\t Loss: {:.5f}.
                '''.format((t1-t0), lr, iteration, loss_c.item(), loss_four_corners.item(), loss_border.item(), loss_rec.item(),
                loss_c.item() + loss_four_corners.item() + loss_border.item() + loss_rec.item(),))

        if args.visdom:
            update_vis_plot(iteration, loss_c.item(), loss_four_corners.item(), loss_border.item(), loss_rec.item(),
                            iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/' + args.save_folder + 'ssd' + 
            str(args.input_size) + '_' + repr(iteration) + '.pth')

    torch.save(ssd_net.state_dict(),
               'weights/' + args.save_folder + '' + args.dataset + str(args.input_size) + '.pth')


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
        Y=torch.zeros((1, 5)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, conf, four_corners, border, rec, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 5)).cpu() * iteration,
        Y=torch.Tensor([conf, four_corners, border, rec, conf + four_corners + border + rec]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 5)).cpu(),
            Y=torch.Tensor([conf, four_corners, border, rec, conf + four_corners + border + rec]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
