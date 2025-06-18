import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import two_stage_end2end, change_cfg_for_ssd512
import os
import numpy as np

from layers.modules import ProposalTargetLayer_offset
# https://github.com/longcw/RoIAlign.pytorch
from roi_align.crop_and_resize import CropAndResizeFunction


def to_varabile(tensor, requires_grad=False, is_cuda=True):
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


# 判断a矩形是否完全包含b矩形
def a_include_b(a_bbox, b_bbox):
    include_or_not = False
    a_xmin, a_ymin, a_xmax, a_ymax = a_bbox
    b_xmin, b_ymin, b_xmax, b_ymax = b_bbox
    if (b_xmin >= a_xmin).cpu().numpy() and (b_ymin >= a_ymin).cpu().numpy()\
            and (b_xmax <= a_xmax).cpu().numpy() and (b_ymax <= a_ymax).cpu().numpy():
        include_or_not = True

    return include_or_not


class SSD_coarse_to_fine(nn.Module):
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

    def __init__(self, phase, size, size_2, base, extras, head, base_2, head_2, num_classes, expand_num):
        super(SSD_coarse_to_fine, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = two_stage_end2end
        if size == 512:
            self.cfg = change_cfg_for_ssd512(self.cfg)
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.priorbox_2 = PriorBox_2(self.cfg)
        with torch.no_grad():
            self.priors_2 = Variable(self.priorbox_2.forward())
        self.size = size
        self.size_2 = size_2
        self.expand_num = expand_num

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

        # SSD network
        self.vgg_2 = nn.ModuleList(base_2)

        self.loc_2 = nn.ModuleList(head_2[0])
        self.conf_2 = nn.ModuleList(head_2[1])
        self.four_corners_2 = nn.ModuleList(head_2[2])

        if phase == 'test':
            self.softmax_2 = nn.Softmax(dim=-1)
            self.detect_2 = Detect_four_corners(num_classes, 0, 200, 0.01, 0.45)

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
        sources = list()
        loc = list()
        conf = list()

        sources_2 = list()
        loc_2 = list()
        conf_2 = list()
        four_corners_2 = list()

        # apply vgg up to conv1_1 relu
        # TODO: may be conv1_1 features
        for k in range(2):
            x = self.vgg[k](x)
            if k == 1:
                # conv1_1 feature relu
                conv1_1_feat = x

        # apply vgg up to conv4_3 relu
        for k in range(2, 23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c, h, s, o) in zip(sources, self.loc, self.conf_2):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # [num, num_classes, top_k, 10]
        rpn_rois = self.detect(
            loc.view(loc.size(0), -1, 4),  # loc preds
            self.softmax(conf.view(conf.size(0), -1,
                                   self.num_classes)),  # conf preds
            self.priors.cuda()  # default boxes 这个地方按照之前会有重大bug,参数分布在不同GPU上
        )

        # 解除这部分的可导
        rpn_rois = rpn_rois.detach()

        # roi align or roi warping
        crop_height = self.size_2
        crop_width = self.size_2
        is_cuda = torch.cuda.is_available()

        if self.phase == 'train':
            # rpn_rois: [num, num_classes, top_k, 10]
            # rois: [num, num_gt, 6], 6: IOU with GT, bbox(4), max iou with GT or not
            # target: [num, num_gt, 22], 10: bbox(4), has_lp, size(2), offset(2),
            # lp_bbox(4), lp_four_points(8), label

            # rois和target最外层是list, 里面是tensor,这样可以确保里面的tensor维度不同
            proposal_target_offset = ProposalTargetLayer_offset()
            rois = proposal_target_offset(rpn_rois, targets, self.expand_num)

            gt_new = torch.empty(0)
            boxes_data_list = []
            box_index_data_list = []
            for idx in range(len(rois)):
                num_gt = targets[idx].shape[0]

                # 获取所有GT车牌的位置
                targets_tensor = targets[idx]
                # car_center_x = (targets_tensor[:, 0].unsqueeze(1) + targets_tensor[:, 2].unsqueeze(1)) / 2.0
                # car_center_y = (targets_tensor[:, 1].unsqueeze(1) + targets_tensor[:, 3].unsqueeze(1)) / 2.0
                # car_center = torch.cat((car_center_x, car_center_y), 1)
                # lp_center = car_center + targets_tensor[:, 7:9]
                # lp_bbox = torch.cat((lp_center - targets_tensor[:, 5:7]/2, lp_center + targets_tensor[:, 5:7]/2), 1)
                lp_bbox = targets_tensor[:, 9:13]

                # 获取车牌的四点坐标
                lp_four_points = targets_tensor[:, 13:21]

                # 获取在rois中的车牌GT,并且根据rois的左上角调整成新的车牌GT
                rois_squeeze = rois[idx][:num_gt, 1:-1]

                a_include_b_list = []
                for i in range(num_gt):
                    a_include_b_list.append(a_include_b(rois_squeeze[i, :], lp_bbox[i, :]))

                has_lp_list = []
                for i in range(num_gt):
                    has_lp_list.append(targets_tensor[i, 4].cpu().numpy() > 0)

                gt_in_rois_list = np.array(a_include_b_list) + 0 & np.array(has_lp_list) + 0
                gt_in_rois_tensor = torch.tensor(gt_in_rois_list).type(torch.uint8).bool()
                rois_squeeze = rois_squeeze[gt_in_rois_tensor, :]
                lp_bbox = lp_bbox[gt_in_rois_tensor, :]
                lp_four_points = lp_four_points[gt_in_rois_tensor, :]

                if rois_squeeze.shape[0] > 0:
                    # 调整车牌GT bbox
                    rois_top_left = rois_squeeze[:, :2].repeat(1, 2)
                    rois_width = rois_squeeze[:, 2] - rois_squeeze[:, 0]
                    rois_height = rois_squeeze[:, 3] - rois_squeeze[:, 1]
                    rois_size = torch.cat((rois_width.unsqueeze(1), rois_height.unsqueeze(1)), 1).repeat(1, 2)
                    gt_bbox = (lp_bbox - rois_top_left) / rois_size

                    # 新的车牌四点
                    rois_top_left_2 = rois_squeeze[:, :2].repeat(1, 4)
                    rois_size_2 = torch.cat((rois_width.unsqueeze(1), rois_height.unsqueeze(1)), 1).repeat(1, 4)
                    gt_four_points = (lp_four_points - rois_top_left_2) / rois_size_2

                    # GT label
                    gt_label = torch.zeros((gt_bbox.shape[0], 1))

                    # is valid,说明这个gt是有效的,因为后面为了迎合多GPU合并必须有输出的情况,后面会伪造一些is not valid的数据
                    # TODO: 这是不太友好的做法
                    gt_valid = torch.ones((gt_bbox.shape[0], 1))

                    # concat
                    gt_cur = torch.cat((gt_bbox, gt_four_points, gt_label, gt_valid), 1)
                    gt_new = torch.cat((gt_new, gt_cur), 0)

                    # 按照损失创造第二个网络的GT,其中gt_2的list要跟后面的crops_torch的n一致,所以用for循环
                    for gt_idx in range(gt_cur.shape[0]):
                        box_index_data_list.append(idx)  # 当前图片的idx

                        boxes_data = torch.zeros(rois_squeeze.shape)
                        boxes_data[:, 0] = rois_squeeze[:, 1]
                        boxes_data[:, 1] = rois_squeeze[:, 0]
                        boxes_data[:, 2] = rois_squeeze[:, 3]
                        boxes_data[:, 3] = rois_squeeze[:, 2]
                        boxes_data_list.append(boxes_data[gt_idx, :].cpu().numpy())  # 当前的区域

            if gt_new.shape[0] > 0:
                # 这是将车作为roi的做法
                # Define the boxes ( crops )
                # box = [y1/heigth , x1/width , y2/heigth , x2/width]
                boxes_data = torch.FloatTensor(boxes_data_list)

                # Create an index to say which box crops which image
                box_index_data = torch.IntTensor(box_index_data_list)

                # Create batch of images
                image_data = conv1_1_feat

                # Convert from numpy to Variables
                # image feature这部分还是需要可导的,参见ROIAlign源程序,训练时需要可导,测试时不需要可导
                image_torch = to_varabile(image_data, is_cuda=is_cuda, requires_grad=True)
                boxes = to_varabile(boxes_data, is_cuda=is_cuda, requires_grad=False)
                box_index = to_varabile(box_index_data, is_cuda=is_cuda, requires_grad=False)

                # Crops and resize bbox1 from img1 and bbox2 from img2
                # n*64*crop_height*crop_width
                crops_torch = CropAndResizeFunction.apply(image_torch, boxes, box_index, crop_height, crop_width, 0)

                # 第二个网络!!!!!!!!!!!!!!!!!!!!!!!!!!
                x_2 = crops_torch

                for k in range(4):
                    x_2 = self.vgg_2[k](x_2)
                sources_2.append(x_2)

                for k in range(4, 9):
                    x_2 = self.vgg_2[k](x_2)
                sources_2.append(x_2)

                for k in range(9, 14):
                    x_2 = self.vgg_2[k](x_2)
                sources_2.append(x_2)

                # apply multibox head to source layers
                for (x_2, l_2, c_2, f_2) in zip(sources_2, self.loc_2, self.conf_2, self.four_corners_2):
                    loc_2.append(l_2(x_2).permute(0, 2, 3, 1).contiguous())
                    conf_2.append(c_2(x_2).permute(0, 2, 3, 1).contiguous())
                    four_corners_2.append(f_2(x_2).permute(0, 2, 3, 1).contiguous())

                loc_2 = torch.cat([o.view(o.size(0), -1) for o in loc_2], 1)
                conf_2 = torch.cat([o.view(o.size(0), -1) for o in conf_2], 1)
                four_corners_2 = torch.cat([o.view(o.size(0), -1) for o in four_corners_2], 1)

            # 如果loc_2还是list,说明gt_new是没有的,第二个网络的预测和GT都为空
            if isinstance(loc_2, list):
                output = (
                    loc.view(loc.size(0), -1, 4),
                    conf.view(conf.size(0), -1, self.num_classes),
                    self.priors,
                    has_lp.view(has_lp.size(0), -1, 1),
                    size_lp.view(size_lp.size(0), -1, 2),
                    offset.view(offset.size(0), -1, 2),
                    # 第二个网络 TODO: 这是非常不友好的做法
                    torch.zeros(1, self.priors_2.shape[0], 4),
                    torch.zeros(1, self.priors_2.shape[0], 2),
                    self.priors_2,
                    torch.zeros(1, self.priors_2.shape[0], 8),
                    torch.zeros(1, 14)  # 最后一位为0表示这个GT not valid
                )
            else:
                output = (
                    loc.view(loc.size(0), -1, 4),
                    conf.view(conf.size(0), -1, self.num_classes),
                    self.priors,
                    # 第二个网络
                    loc_2.view(loc_2.size(0), -1, 4),
                    conf_2.view(conf_2.size(0), -1, self.num_classes),
                    self.priors_2,
                    four_corners_2.view(four_corners_2.size(0), -1, 8),
                    gt_new
                )

        elif self.phase == 'test':
            has_lp_th = 0.5
            th = 0.6
            # 包括车和车牌的检测结果
            output = torch.zeros(1, 3, 200, 13)
            # 存储车的检测结果
            output[0, 1, :, :5] = rpn_rois[0, 1, :, :5]

            # 这里把是否有车牌也考虑进来,有车并且有车牌的才去检测车牌
            rois_idx = (rpn_rois[0, 1, :, 0] > th) & (rpn_rois[0, 1, :, 5] > has_lp_th)
            matches = rpn_rois[0, 1, rois_idx, :]
            if matches.shape[0] == 0:
                return output

            # 针对matches中offset,size以及扩大倍数在车内扩大
            car_center = (matches[:, [1, 2]] + matches[:, [3, 4]]) / 2
            lp_center = car_center + matches[:, [8, 9]]
            lp_bbox_top_left = lp_center - matches[:, [6, 7]] / 2 * self.expand_num
            lp_bbox_bottom_right = lp_center + matches[:, [6, 7]] / 2 * self.expand_num
            lp_bbox = torch.cat((lp_bbox_top_left, lp_bbox_bottom_right), 1)
            # 将扩大后的车牌区域限制在图片内
            lp_bbox = torch.max(lp_bbox, torch.zeros(lp_bbox.shape))
            lp_bbox = torch.min(lp_bbox, torch.ones(lp_bbox.shape))
            # 将扩大后的车牌区域限制在检测到的车内
            lp_bbox = torch.max(lp_bbox, matches[:, 1:3].repeat(1, 2))
            lp_bbox = torch.min(lp_bbox, matches[:, 3:5].repeat(1, 2))

            # [num_car, 4]
            rois_squeeze = lp_bbox

            # 这是将车作为roi的做法
            # Define the boxes ( crops )
            # box = [y1/heigth , x1/width , y2/heigth , x2/width]
            boxes_data = torch.zeros(rois_squeeze.shape)
            boxes_data[:, 0] = rois_squeeze[:, 1]
            boxes_data[:, 1] = rois_squeeze[:, 0]
            boxes_data[:, 2] = rois_squeeze[:, 3]
            boxes_data[:, 3] = rois_squeeze[:, 2]

            # Create an index to indicate which box crops which image
            box_index_data = torch.IntTensor(range(boxes_data.shape[0]))

            # Create a batch of 2 images
            # 这个地方非常关键,需要repeat,不然后面的feature全是0 !!!!!!!!!!!!!!!
            image_data = conv1_1_feat.repeat(rois_squeeze.shape[0], 1, 1, 1)

            # Convert from numpy to Variables
            # image feature这部分还是需要可导的
            image_torch = to_varabile(image_data, is_cuda=is_cuda, requires_grad=False)
            boxes = to_varabile(boxes_data, is_cuda=is_cuda, requires_grad=False)
            box_index = to_varabile(box_index_data, is_cuda=is_cuda, requires_grad=False)

            # Crops and resize bbox1 from img1 and bbox2 from img2
            # n*64*crop_height*crop_width
            crops_torch = CropAndResizeFunction.apply(image_torch, boxes, box_index, crop_height, crop_width, 0)

            # Visualize the crops
            # print(crops_torch.data.size())
            # crops_torch_data = crops_torch.data.cpu().numpy().transpose(0, 2, 3, 1)
            # import matplotlib.pyplot as plt
            # for m in range(rois_squeeze.shape[0]):
            #     fig = plt.figure()
            #     currentAxis = plt.gca()
            #     # pt = gt_2[m][0, :4].cpu().numpy() * self.size_2
            #     # coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            #     # currentAxis.add_patch(plt.Rectangle(*coords, fill=False))
            #     plt.imshow(crops_torch_data[m, :, :, 33])
            #     plt.show()

            # 第二个网络!!!!!!!!!!!!!!!!!!!!!!!!!!
            x_2 = crops_torch

            for k in range(4):
                x_2 = self.vgg_2[k](x_2)
            sources_2.append(x_2)

            for k in range(4, 9):
                x_2 = self.vgg_2[k](x_2)
            sources_2.append(x_2)

            for k in range(9, 14):
                x_2 = self.vgg_2[k](x_2)
            sources_2.append(x_2)

            # apply multibox head to source layers
            for (x_2, l_2, c_2, f_2) in zip(sources_2, self.loc_2, self.conf_2, self.four_corners_2):
                loc_2.append(l_2(x_2).permute(0, 2, 3, 1).contiguous())
                conf_2.append(c_2(x_2).permute(0, 2, 3, 1).contiguous())
                four_corners_2.append(f_2(x_2).permute(0, 2, 3, 1).contiguous())

            loc_2 = torch.cat([o.view(o.size(0), -1) for o in loc_2], 1)
            conf_2 = torch.cat([o.view(o.size(0), -1) for o in conf_2], 1)
            four_corners_2 = torch.cat([o.view(o.size(0), -1) for o in four_corners_2], 1)

            output_2 = self.detect_2(
                loc_2.view(loc_2.size(0), -1, 4),
                self.softmax_2(conf_2.view(conf_2.size(0), -1,
                                            self.num_classes)),
                self.priors_2.cuda(),
                four_corners_2.view(four_corners_2.size(0), -1, 8)
            )
            
            # 这种方法是综合所有车里面的车牌检测结果,然后只选取所有结果的前200个
            # (num_car, 200, 13)
            # output_2_pos = output_2[:, 1, :, :]
            # # (num_car, 2)
            # rois_size = rois_squeeze[:, 2:4] - rois_squeeze[:, :2]
            # rois_top_left = rois_squeeze[:, :2]
            # # (num_car, 200, 12)
            # rois_size_expand = rois_size.repeat(1, 6).unsqueeze(1).repeat(1, 200, 1)
            # # (num_car, 200, 12)
            # rois_top_left_expand = rois_top_left.repeat(1, 6).unsqueeze(1).repeat(1, 200, 1)
            # # (num_car, 200, 12)
            # output_2_pos[:, :, 1:] = output_2_pos[:, :, 1:] * rois_size_expand + rois_top_left_expand
            # # (num_car*200, 13)
            # output_2_pos_squeeze = output_2_pos.reshape(-1, output_2_pos.shape[2])
            # _, indices = output_2_pos_squeeze[:, 0].sort(descending=True)
            # output_2_pos_squeeze_sorted = output_2_pos_squeeze[indices, :]
            # # (1, 2, 200, 13)
            # results_2 = output_2_pos_squeeze_sorted[:200, :].unsqueeze(0).unsqueeze(1).repeat(1, 2, 1, 1)

            # 这种方法是每辆车里面只选conf最大的车牌
            # (num_car, 13)
            output_2_pos = output_2[:, 1, 0, :]
            # (num_car, 2)
            rois_size = rois_squeeze[:, 2:4] - rois_squeeze[:, :2]
            rois_top_left = rois_squeeze[:, :2]
            # (num_car, 12)
            rois_size_expand = rois_size.repeat(1, 6)
            # (num_car, 12)
            rois_top_left_expand = rois_top_left.repeat(1, 6)
            # (num_car, 12)
            output_2_pos[:, 1:] = output_2_pos[:, 1:] * rois_size_expand + rois_top_left_expand

            # 存储车牌的检测结果
            num_car = output_2_pos.shape[0]
            output[0, 2, :num_car, :] = output_2_pos

            # 存储expand区域的结果,放在车后面,并设置flag
            output[0, 1, :num_car, 5:9] = lp_bbox
            output[0, 1, :num_car, 9] = 1

            return output
        else:
            print("ERROR: Phase: " + self.phase + " not recognized")
            return

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
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

            # # 固定第一层权重
            # if in_channels == 3:
            #     for p in conv2d.parameters():
            #         p.requires_grad = False

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


def vgg_2(cfg, i, batch_norm=False):
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


def multibox(vgg, extra_layers, cfg, num_classes, vgg_2, cfg_2):
    loc_layers = []
    conf_layers = []
    has_lp_layers = []
    size_lp_layers = []
    offset_layers = []
    vgg_source = [21, -2]

    loc_layers_2 = []
    conf_layers_2 = []
    four_corners_layers_2 = []
    vgg_source_2 = [2, 7, 12]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
        has_lp_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * 1, kernel_size=3, padding=1)]
        size_lp_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * 2, kernel_size=3, padding=1)]
        offset_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * 2, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
        has_lp_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * 1, kernel_size=3, padding=1)]
        size_lp_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * 2, kernel_size=3, padding=1)]
        offset_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * 2, kernel_size=3, padding=1)]

    for k, v in enumerate(vgg_source_2):
        loc_layers_2 += [nn.Conv2d(vgg_2[v].out_channels,
                                 cfg_2[k] * 4, kernel_size=3, padding=1)]
        conf_layers_2 += [nn.Conv2d(vgg_2[v].out_channels,
                        cfg_2[k] * num_classes, kernel_size=3, padding=1)]
        four_corners_layers_2 += [nn.Conv2d(vgg_2[v].out_channels,
                                          cfg_2[k] * 8, kernel_size=3, padding=1)]

    return vgg, extra_layers, (loc_layers, conf_layers, has_lp_layers, size_lp_layers, offset_layers),\
           vgg_2, (loc_layers_2, conf_layers_2, four_corners_layers_2)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '56': [512, 512, 'M', 512, 512, 'M', 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],
    '56': [6, 6, 6],
    '512': [4, 6, 6, 6, 6, 4, 4],
}


def build_ssd(phase, size=300, size_2=56, num_classes=21, expand_num=3):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300 and size != 512:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 SSD512 (size=300 or size=512) is supported!")
        return
    base_, extras_, head_, base_2_, head_2_ = multibox(vgg(base[str(size)], 3),
                                                      add_extras(extras[str(size)], size, 1024),
                                                      mbox[str(size)],
                                                      num_classes,
                                                      vgg_2(base[str(size_2)], 64),
                                                      mbox[str(size_2)]
                                                      )
    return SSD_coarse_to_fine(phase, size, size_2, base_, extras_, head_, base_2_, head_2_, num_classes, expand_num)
