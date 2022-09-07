import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import carplate, change_cfg_for_ssd512, CHARS
import os
import time
from roi_align.crop_and_resize import CropAndResizeFunction
import PVAM.LPR_inves_model as PVAM


def to_varabile(tensor, requires_grad=False, is_cuda=True):
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class SSD_PVAM(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes,
                reg_size=(96,32), lpr_max_len=8, class_num=len(CHARS)):
        super(SSD_PVAM, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = carplate
        if size == 512:
            self.cfg = change_cfg_for_ssd512(self.cfg)
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.conf = nn.ModuleList(head[0])
        self.loc = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect()
            self.perspective_trans_function = PerspectiveTrans()

        self.reg_size = reg_size
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num

        self.lpr_model = PVAM.LPR_model_trans_baseline(64, class_num, imgW=reg_size[0], imgH=reg_size[1],  K=lpr_max_len, isSeqModel=True, head=4, inner=512, model_dim=256).cuda()

    def forward(self, x, targets):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        _t = {'im_detect': Timer(), 'misc': Timer()}
        _t['im_detect'].tic()
        sources = list()
        conf = list()
        loc = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)
            if k == 0:
                conv1_1_feat = x
            elif k == 7:
                conv2_2_feat = x
            elif k == 14:
                conv3_3_feat = x
            elif k == 21:
                conv4_3_feat = x

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
            if k == 29:
                conv5_3_feat = x

        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, c, l) in zip(sources, self.conf, self.loc):
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())

        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        detect_time = _t['im_detect'].toc(average=False)
        # print("ssd forward detect_time: " + str(detect_time))

        if self.phase == "test":
            rpn_rois = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                 # default boxes
            )
            # 解除这部分的可导
            rpn_rois = rpn_rois.detach()
            # [num, num_classes, top_k, 5]
            lp_bbox = (rpn_rois[0][1][0][1:5]).unsqueeze(0)
            lp_bbox = torch.clamp(lp_bbox, min=0.0, max=1.0)
            boxes_data_list = []
            box_index_data_list = []
            # from pred bbox
            boxes_data = torch.zeros(lp_bbox.shape)
            boxes_data[:, 0] = lp_bbox[:, 1]
            boxes_data[:, 1] = lp_bbox[:, 0]
            boxes_data[:, 2] = lp_bbox[:, 3]
            boxes_data[:, 3] = lp_bbox[:, 2]
            box_index_data_list.append(0)  # 当前图片的idx
            boxes_data_list.append(boxes_data[0, :].cpu().numpy())  # 每个车牌的bbox
            boxes_data = torch.FloatTensor(boxes_data_list)
            box_index_data = torch.IntTensor(box_index_data_list)
            image_data = conv1_1_feat
            image_data = to_varabile(image_data, is_cuda=True, requires_grad=True)
            boxes_data = to_varabile(boxes_data, is_cuda=True, requires_grad=False)
            box_index_data = to_varabile(box_index_data, is_cuda=True, requires_grad=False)
            # n*64*crop_height*crop_width
            x_rec = CropAndResizeFunction.apply(image_data, boxes_data, box_index_data, self.reg_size[1], self.reg_size[0], 0)
            # print('x_rec: ', x_rec.shape)
            # import cv2
            # for i in range(x_rec.shape[0]):
            #     cv2.imshow(str(i), x_rec[i, 33:36, :, :].data.cpu().numpy().transpose(1, 2, 0))
            # cv2.waitKey()
        else:
            # print("targets: ", targets)
            boxes_data_list = []
            box_index_data_list = []
            # from GT
            for idx in range(len(targets)):
                # num_gt = targets[idx].shape[0]
                targets_tensor = targets[idx]
                lp_bbox = targets_tensor[:, :4]
                boxes_data = torch.zeros(lp_bbox.shape)
                boxes_data[:, 0] = lp_bbox[:, 1]
                boxes_data[:, 1] = lp_bbox[:, 0]
                boxes_data[:, 2] = lp_bbox[:, 3]
                boxes_data[:, 3] = lp_bbox[:, 2]
                for gt_idx in range(targets_tensor.shape[0]):
                    box_index_data_list.append(idx)  # 当前图片的idx
                    boxes_data_list.append(boxes_data[gt_idx, :].cpu().numpy())  # 每个车牌的bbox
            
            boxes_data = torch.FloatTensor(boxes_data_list)
            box_index_data = torch.IntTensor(box_index_data_list)
            image_data = conv1_1_feat
            image_data = to_varabile(image_data, is_cuda=True, requires_grad=True)
            boxes_data = to_varabile(boxes_data, is_cuda=True, requires_grad=False)
            box_index_data = to_varabile(box_index_data, is_cuda=True, requires_grad=False)
            # n*64*crop_height*crop_width
            x_rec = CropAndResizeFunction.apply(image_data, boxes_data, box_index_data, self.reg_size[1], self.reg_size[0], 0)
            # print("crops_torch: ", x_rec.shape)
            # import cv2
            # for i in range(x_rec.shape[0]):
            #     cv2.imshow(str(i), x_rec[i, 4:7, :, :].data.cpu().numpy().transpose(1, 2, 0))
            # cv2.waitKey()

        _, preds = self.lpr_model(x_rec)
        # print('preds', preds.shape)

        if self.phase == "test":
            detection_output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                 # default boxes
            )
            recognition_output = preds
        else:
            detection_output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
            recognition_output = preds
        return detection_output, recognition_output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage), strict=False)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, size, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    # SSD512 need add two more Conv layer
    if size == 512:
        layers += [nn.Conv2d(in_channels, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1)]
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    conf_layers = []
    loc_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * 4, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * 4, kernel_size=3, padding=1)]
    return vgg, extra_layers, (conf_layers, loc_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],
    '512': [4, 6, 6, 6, 6, 4, 4],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300 and size != 512:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 SSD512 (size=300 or size=512) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], size, 1024),
                                     mbox[str(size)], num_classes)
    return SSD_PVAM(phase, size, base_, extras_, head_, num_classes)
