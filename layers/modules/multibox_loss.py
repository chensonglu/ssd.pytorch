# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import voc as cfg
from ..box_utils import match, match_offset, match_four_corners, match_only_four_corners, log_sum_exp
from ..box_utils import decode, decode_four_corners, decode_four_corners_TextBoxesPlusPlus
from ..box_utils import match_ious, bbox_overlaps_iou, bbox_overlaps_giou, bbox_overlaps_diou, bbox_overlaps_ciou
from ..functions.detection import Detect_four_corners


class IouLoss(nn.Module):
    def __init__(self, pred_mode='Center', size_sum=True, variances=None, losstype='Ciou'):
        super(IouLoss, self).__init__()
        self.size_sum = size_sum
        self.pred_mode = pred_mode
        self.variances = variances
        self.loss = losstype
    def forward(self, loc_p, loc_t, prior_data):
        num = loc_p.shape[0]

        if self.pred_mode == 'Center':
            decoded_boxes = decode(loc_p, prior_data, self.variances)
        else:
            decoded_boxes = loc_p
        if self.loss == 'Iou':
            loss = torch.sum(1.0 - bbox_overlaps_iou(decoded_boxes, loc_t))
        else:
            if self.loss == 'Giou':
                loss = torch.sum(1.0 - bbox_overlaps_giou(decoded_boxes, loc_t))
            else:
                if self.loss == 'Diou':
                    loss = torch.sum(1.0 - bbox_overlaps_diou(decoded_boxes, loc_t))
                else:
                    loss = torch.sum(1.0 - bbox_overlaps_ciou(decoded_boxes, loc_t))

        if self.size_sum:
            loss = loss
        else:
            loss = loss / num
        return loss


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True, loss_name='SmoothL1'):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.loss = loss_name
        self.pred_mode = 'Center'
        self.iou_loss = IouLoss(pred_mode=self.pred_mode, size_sum=True, variances=self.variance, losstype=self.loss)
        if self.loss != 'SmoothL1' or self.loss != 'Ciou':
            assert Exception("loss name must be SmoothL1 or Ciou")

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            if self.loss == 'SmoothL1':
                match(self.threshold, truths, defaults, self.variance, labels,
                    loc_t, conf_t, idx)
            else:
                match_ious(self.threshold, truths, defaults, self.variance, labels,
                    loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        if self.loss == 'SmoothL1':
            loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        else:
            ciou_priors = priors.data.unsqueeze(0).expand_as(loc_data)
            loss_l = self.iou_loss(loc_p, loc_t, ciou_priors[pos_idx].view(-1, 4))

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum().double()
        if self.loss == 'SmoothL1':
            loss_l = loss_l.double()
        else:
            loss_l = 16.0 * loss_l.double()
        loss_c = loss_c.double()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c


class MultiBoxLoss_offset(nn.Module):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss_offset, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds, has_lp preds, size_lp preds, offset preds
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
                has_lp shape: torch.size(batch_size,num_priors,1)
                size_lp shape: torch.size(batch_size,num_priors,2)
                offset shape: torch.size(batch_size,num_priors,2)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,10] (last idx is the label).
        """
        loc_data, conf_data, priors, has_lp_data, size_lp_data, offset_data = predictions

        num = loc_data.size(0)  # batchsize
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        has_lp_t = torch.LongTensor(num, num_priors)
        size_lp_t = torch.Tensor(num, num_priors, 2)
        offset_t = torch.Tensor(num, num_priors, 2)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match_offset(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, has_lp_t, size_lp_t, offset_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            has_lp_t = has_lp_t.cuda()
            size_lp_t = size_lp_t.cuda()
            offset_t = offset_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        has_lp_t = Variable(has_lp_t, requires_grad=False)
        size_lp_t = Variable(size_lp_t, requires_grad=False)
        offset_t = Variable(offset_t, requires_grad=False)

        pos = conf_t > 0  # label > 0 for bbox regression, [batch,num_priors]
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1), only for positive prior
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Size Loss (Smooth L1), only for positive prior and has lp
        # Shape: [batch,num_priors,2]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(size_lp_data)
        has_lp_idx = has_lp_t.unsqueeze(pos.dim()).expand_as(size_lp_data).view(num, num_priors, 2) > 0

        size_lp_p = size_lp_data[pos_idx & has_lp_idx].view(-1, 2)
        size_lp_t = size_lp_t[pos_idx & has_lp_idx].view(-1, 2)
        
        loss_size_lp = F.smooth_l1_loss(size_lp_p, size_lp_t, size_average=False)

        # Offset Loss (Smooth L1), only for positive prior and has lp
        # Shape: [batch,num_priors,2]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(offset_data)
        has_lp_idx = has_lp_t.unsqueeze(pos.dim()).expand_as(offset_data).view(num, num_priors, 2) > 0

        offset_p = offset_data[pos_idx & has_lp_idx].view(-1, 2)
        offset_t = offset_t[pos_idx & has_lp_idx].view(-1, 2)

        loss_offset = F.smooth_l1_loss(offset_p, offset_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # 参考https://www.cnblogs.com/JeasonIsCoding/p/10171201.html推导具体公式
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        # 根据两次sort可以找出元素在升序或者降序中的位置,参考https://blog.csdn.net/LXX516/article/details/78804884
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]

        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Has Carplate Loss (Binary Cross Entropy), Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(has_lp_data)
        neg_idx = neg.unsqueeze(2).expand_as(has_lp_data)

        has_lp_p = has_lp_data[(pos_idx+neg_idx).gt(0)].view(-1, 1).float()
        has_lp_t = has_lp_t[(pos+neg).gt(0)].view(-1, 1).float()

        loss_has_lp = F.binary_cross_entropy(torch.sigmoid(has_lp_p), has_lp_t, size_average=False)

        # Sum of losses
        N = num_pos.data.sum().double()
        loss_l = loss_l.double()
        loss_c = loss_c.double()
        loss_size_lp = loss_size_lp.double()
        loss_offset = loss_offset.double()
        loss_has_lp = loss_has_lp.double()
        loss_l /= N
        loss_c /= N
        loss_size_lp /= N
        loss_offset /= N
        loss_has_lp /= N
        return loss_l, loss_c, loss_size_lp, loss_offset, loss_has_lp


class MultiBoxLoss_four_corners(nn.Module):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss_four_corners, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
                four_corners shape: torch.size(batch_size,num_priors,8)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,13] (last idx is the label).
        """
        loc_data, conf_data, priors, four_corners_data = predictions

        num = loc_data.size(0)  # batchsize
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        four_corners_t = torch.Tensor(num, num_priors, 8)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match_four_corners(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, four_corners_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            four_corners_t = four_corners_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        four_corners_t = Variable(four_corners_t, requires_grad=False)

        pos = conf_t > 0  # label > 0 for bbox regression, [batch,num_priors]
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1), only for positive prior
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Corner Loss (Smooth L1), only for positive prior
        # Shape: [batch,num_priors,8]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(four_corners_data)

        four_corners_p = four_corners_data[pos_idx].view(-1, 8)
        four_corners_t = four_corners_t[pos_idx].view(-1, 8)

        loss_four_corners = F.smooth_l1_loss(four_corners_p, four_corners_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # 参考https://www.cnblogs.com/JeasonIsCoding/p/10171201.html推导具体公式
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        # 根据两次sort可以找出元素在升序或者降序中的位置,参考https://blog.csdn.net/LXX516/article/details/78804884
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]

        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses
        N = num_pos.data.sum().double()
        loss_l = loss_l.double()
        loss_c = loss_c.double()
        loss_four_corners = loss_four_corners.double()
        loss_l /= N
        loss_c /= N
        loss_four_corners /= N
        return loss_l, loss_c, loss_four_corners


class MultiBoxLoss_four_corners_with_border(nn.Module):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss_four_corners_with_border, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
                four_corners shape: torch.size(batch_size,num_priors,8)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,13] (last idx is the label).
        """
        loc_data, conf_data, priors, four_corners_data = predictions
        num = loc_data.size(0)  # batchsize
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        four_corners_t = torch.Tensor(num, num_priors, 8)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match_four_corners(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, four_corners_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            four_corners_t = four_corners_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        four_corners_t = Variable(four_corners_t, requires_grad=False)

        pos = conf_t > 0  # label > 0 for bbox regression, [batch,num_priors]
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1), only for positive prior
        # Shape: [batch,num_priors,4]
        pos_idx_loc = pos.unsqueeze(pos.dim()).expand_as(loc_data)  # [batch,num_priors,4]

        loc_p = loc_data[pos_idx_loc].view(-1, 4)
        loc_t = loc_t[pos_idx_loc].view(-1, 4)

        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Corner Loss (Smooth L1), only for positive prior
        # Shape: [batch,num_priors,8]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(four_corners_data)  # [batch,num_priors,8]

        four_corners_p = four_corners_data[pos_idx].view(-1, 8)
        four_corners_t = four_corners_t[pos_idx].view(-1, 8)

        loss_four_corners = F.smooth_l1_loss(four_corners_p, four_corners_t, size_average=False)

        # Border Loss (Smooth L1), only for positive prior
        # 需要首先decode, 将priors扩展成[batch,num_priors,4], 然后根据pos得到[batch,num_pos,4]
        priors_pos = priors.unsqueeze(0).expand_as(loc_data)[pos_idx_loc].view(-1, 4)
        decoded_boxes = decode(loc_p, priors_pos, self.variance)  # [xmin, ymin, xmax, ymax]
        decoded_four_points = decode_four_corners(four_corners_p, priors_pos, self.variance)  # [x_top_left, y_top_left, ...]

        # Border Loss (Smooth L1), only for positive prior 
        # 4 border losses
        loss_border_left = F.smooth_l1_loss(torch.min(decoded_four_points[:, 0], decoded_four_points[:, 6]) / self.variance[0],
                                            decoded_boxes[:, 0] / self.variance[0],
                                            size_average=False)
        loss_border_top = F.smooth_l1_loss(torch.min(decoded_four_points[:, 1], decoded_four_points[:, 3]) / self.variance[0],
                                           decoded_boxes[:, 1] / self.variance[0],
                                           size_average=False)
        loss_border_right = F.smooth_l1_loss(torch.max(decoded_four_points[:, 2], decoded_four_points[:, 4]) / self.variance[0],
                                             decoded_boxes[:, 2] / self.variance[0],
                                             size_average=False)
        loss_border_bottom = F.smooth_l1_loss(torch.max(decoded_four_points[:, 5], decoded_four_points[:, 7]) / self.variance[0],
                                              decoded_boxes[:, 3] / self.variance[0],
                                              size_average=False)

        loss_border = loss_border_left + loss_border_top + loss_border_right + loss_border_bottom

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # 参考https://www.cnblogs.com/JeasonIsCoding/p/10171201.html推导具体公式
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        # 根据两次sort可以找出元素在升序或者降序中的位置,参考https://blog.csdn.net/LXX516/article/details/78804884
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]

        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses
        N = num_pos.data.sum().double()
        loss_l = loss_l.double()
        loss_c = loss_c.double()
        loss_four_corners = loss_four_corners.double()
        loss_border = 16.0 * loss_border.double()
        loss_l /= N
        loss_c /= N
        loss_four_corners /= N
        loss_border /= N
        return loss_l, loss_c, loss_four_corners, loss_border


class MultiBoxLoss_four_corners_with_CIoU(nn.Module):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss_four_corners_with_CIoU, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.loss = 'Ciou'
        self.pred_mode = 'Bbox'
        self.iou_loss = IouLoss(pred_mode=self.pred_mode, size_sum=True, variances=self.variance, losstype=self.loss)

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
                four_corners shape: torch.size(batch_size,num_priors,8)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,13] (last idx is the label).
        """
        loc_data, conf_data, priors, four_corners_data = predictions
        num = loc_data.size(0)  # batchsize
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        four_corners_t = torch.Tensor(num, num_priors, 8)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match_four_corners(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, four_corners_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            four_corners_t = four_corners_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        four_corners_t = Variable(four_corners_t, requires_grad=False)

        pos = conf_t > 0  # label > 0 for bbox regression, [batch,num_priors]
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1), only for positive prior
        # Shape: [batch,num_priors,4]
        pos_idx_loc = pos.unsqueeze(pos.dim()).expand_as(loc_data)  # [batch,num_priors,4]

        loc_p = loc_data[pos_idx_loc].view(-1, 4)
        loc_t = loc_t[pos_idx_loc].view(-1, 4)

        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Corner Loss (Smooth L1), only for positive prior
        # Shape: [batch,num_priors,8]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(four_corners_data)  # [batch,num_priors,8]

        four_corners_p = four_corners_data[pos_idx].view(-1, 8)
        four_corners_t = four_corners_t[pos_idx].view(-1, 8)

        loss_four_corners = F.smooth_l1_loss(four_corners_p, four_corners_t, size_average=False)

        # CIoU Loss only for positive prior
        # 需要首先decode, 将priors扩展成[batch,num_priors,4], 然后根据pos得到[batch,num_pos,4]
        priors_pos = priors.unsqueeze(0).expand_as(loc_data)[pos_idx_loc].view(-1, 4)
        decoded_boxes = decode(loc_p, priors_pos, self.variance)  # [xmin, ymin, xmax, ymax]
        decoded_four_points = decode_four_corners(four_corners_p, priors_pos, self.variance)  # [x_top_left, y_top_left, ...]

        left = torch.min(decoded_four_points[:, 0], decoded_four_points[:, 6]).view(-1, 1)
        top = torch.min(decoded_four_points[:, 1], decoded_four_points[:, 3]).view(-1, 1)
        right = torch.max(decoded_four_points[:, 2], decoded_four_points[:, 4]).view(-1, 1)
        bottom = torch.max(decoded_four_points[:, 5], decoded_four_points[:, 7]).view(-1, 1)
        decoded_boxes_from_four_points = torch.cat((left, top, right, bottom), 1)

        loss_border = self.iou_loss(decoded_boxes_from_four_points, decoded_boxes, priors_pos)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # 参考https://www.cnblogs.com/JeasonIsCoding/p/10171201.html推导具体公式
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        # 根据两次sort可以找出元素在升序或者降序中的位置,参考https://blog.csdn.net/LXX516/article/details/78804884
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]

        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses
        N = num_pos.data.sum().double()
        loss_l = loss_l.double()
        loss_c = loss_c.double()
        loss_four_corners = loss_four_corners.double()
        loss_border = 16.0 * loss_border.double()
        loss_l /= N
        loss_c /= N
        loss_four_corners /= N
        loss_border /= N
        return loss_l, loss_c, loss_four_corners, loss_border


class MultiBoxLoss_only_four_corners(nn.Module):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss_only_four_corners, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
                four_corners shape: torch.size(batch_size,num_priors,8)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,13] (last idx is the label).
        """
        conf_data, priors, four_corners_data = predictions

        num = conf_data.size(0)  # batchsize
        priors = priors[:conf_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        four_corners_t = torch.Tensor(num, num_priors, 8)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match_only_four_corners(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, four_corners_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            four_corners_t = four_corners_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        four_corners_t = Variable(four_corners_t, requires_grad=False)

        pos = conf_t > 0  # label > 0 for bbox regression, [batch,num_priors]
        num_pos = pos.sum(dim=1, keepdim=True)

        # Corner Loss (Smooth L1), only for positive prior
        # Shape: [batch,num_priors,8]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(four_corners_data)

        four_corners_p = four_corners_data[pos_idx].view(-1, 8)
        four_corners_t = four_corners_t[pos_idx].view(-1, 8)

        loss_four_corners = F.smooth_l1_loss(four_corners_p, four_corners_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # 参考https://www.cnblogs.com/JeasonIsCoding/p/10171201.html推导具体公式
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        # 根据两次sort可以找出元素在升序或者降序中的位置,参考https://blog.csdn.net/LXX516/article/details/78804884
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]

        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses
        N = num_pos.data.sum().double()
        loss_c = loss_c.double()
        loss_four_corners = loss_four_corners.double()
        loss_c /= N
        loss_four_corners /= N
        return loss_c, loss_four_corners


class MultiBoxLoss_only_four_corners_with_border(nn.Module):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss_only_four_corners_with_border, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
                four_corners shape: torch.size(batch_size,num_priors,8)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,13] (last idx is the label).
        """
        conf_data, priors, four_corners_data = predictions

        num = conf_data.size(0)  # batchsize
        priors = priors[:conf_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        four_corners_t = torch.Tensor(num, num_priors, 8)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match_only_four_corners(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, four_corners_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            four_corners_t = four_corners_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        four_corners_t = Variable(four_corners_t, requires_grad=False)

        pos = conf_t > 0  # label > 0 for bbox regression, [batch,num_priors]
        num_pos = pos.sum(dim=1, keepdim=True)

        # Shape: [batch,num_priors,4]
        pos_idx_loc = pos.unsqueeze(pos.dim()).expand_as(four_corners_data[:,:,:4])
        loc_t = loc_t[pos_idx_loc].view(-1, 4)

        # Corner Loss (Smooth L1), only for positive prior
        # Shape: [batch,num_priors,8]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(four_corners_data)

        four_corners_p = four_corners_data[pos_idx].view(-1, 8)
        four_corners_t = four_corners_t[pos_idx].view(-1, 8)

        loss_four_corners = F.smooth_l1_loss(four_corners_p, four_corners_t, size_average=False)

        # Border Loss (Smooth L1), only for positive prior
        # 需要首先decode, 将priors扩展成[batch,num_priors,4], 然后根据pos得到[batch,num_pos,4]
        priors_pos = priors.unsqueeze(0).expand_as(four_corners_data[:,:,:4])[pos_idx_loc].view(-1, 4)
        decoded_four_points = decode_four_corners(four_corners_p, priors_pos, self.variance)  # [x_top_left, y_top_left, ...]

        # Border Loss (Smooth L1), only for positive prior
        # 4 border losses
        loss_border_left = F.smooth_l1_loss(torch.min(decoded_four_points[:, 0], decoded_four_points[:, 6]) / self.variance[0],
                                            loc_t[:, 0] / self.variance[0],
                                            size_average=False)
        loss_border_top = F.smooth_l1_loss(torch.min(decoded_four_points[:, 1], decoded_four_points[:, 3]) / self.variance[0],
                                           loc_t[:, 1] / self.variance[0],
                                           size_average=False)
        loss_border_right = F.smooth_l1_loss(torch.max(decoded_four_points[:, 2], decoded_four_points[:, 4]) / self.variance[0],
                                             loc_t[:, 2] / self.variance[0],
                                             size_average=False)
        loss_border_bottom = F.smooth_l1_loss(torch.max(decoded_four_points[:, 5], decoded_four_points[:, 7]) / self.variance[0],
                                              loc_t[:, 3] / self.variance[0],
                                              size_average=False)

        loss_border = loss_border_left + loss_border_top + loss_border_right + loss_border_bottom

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # 参考https://www.cnblogs.com/JeasonIsCoding/p/10171201.html推导具体公式
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        # 根据两次sort可以找出元素在升序或者降序中的位置,参考https://blog.csdn.net/LXX516/article/details/78804884
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]

        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses
        N = num_pos.data.sum().double()
        loss_c = loss_c.double()
        loss_four_corners = loss_four_corners.double()
        loss_border = 16.0 * loss_border.double()
        loss_c /= N
        loss_four_corners /= N
        loss_border /= N
        return loss_c, loss_four_corners, loss_border


class MultiBoxLoss_only_four_corners_with_CIoU(nn.Module):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss_only_four_corners_with_CIoU, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.loss = 'Ciou'
        self.pred_mode = 'Bbox'
        self.iou_loss = IouLoss(pred_mode=self.pred_mode, size_sum=True, variances=self.variance, losstype=self.loss)

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
                four_corners shape: torch.size(batch_size,num_priors,8)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,13] (last idx is the label).
        """
        conf_data, priors, four_corners_data = predictions

        num = conf_data.size(0)  # batchsize
        priors = priors[:conf_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        four_corners_t = torch.Tensor(num, num_priors, 8)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match_only_four_corners(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, four_corners_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            four_corners_t = four_corners_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        four_corners_t = Variable(four_corners_t, requires_grad=False)

        pos = conf_t > 0  # label > 0 for bbox regression, [batch,num_priors]
        num_pos = pos.sum(dim=1, keepdim=True)

        # Shape: [batch,num_priors,4]
        pos_idx_loc = pos.unsqueeze(pos.dim()).expand_as(four_corners_data[:,:,:4])
        loc_t = loc_t[pos_idx_loc].view(-1, 4)

        # Corner Loss (Smooth L1), only for positive prior
        # Shape: [batch,num_priors,8]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(four_corners_data)

        four_corners_p = four_corners_data[pos_idx].view(-1, 8)
        four_corners_t = four_corners_t[pos_idx].view(-1, 8)

        loss_four_corners = F.smooth_l1_loss(four_corners_p, four_corners_t, size_average=False)

        # CIoU Loss only for positive prior
        # 需要首先decode, 将priors扩展成[batch,num_priors,4], 然后根据pos得到[batch,num_pos,4]
        priors_pos = priors.unsqueeze(0).expand_as(four_corners_data[:,:,:4])[pos_idx_loc].view(-1, 4)
        decoded_four_points = decode_four_corners(four_corners_p, priors_pos, self.variance)  # [x_top_left, y_top_left, ...]

        left = torch.min(decoded_four_points[:, 0], decoded_four_points[:, 6]).view(-1, 1)
        top = torch.min(decoded_four_points[:, 1], decoded_four_points[:, 3]).view(-1, 1)
        right = torch.max(decoded_four_points[:, 2], decoded_four_points[:, 4]).view(-1, 1)
        bottom = torch.max(decoded_four_points[:, 5], decoded_four_points[:, 7]).view(-1, 1)
        decoded_boxes_from_four_points = torch.cat((left, top, right, bottom), 1)

        loss_border = self.iou_loss(decoded_boxes_from_four_points, loc_t, priors_pos)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # 参考https://www.cnblogs.com/JeasonIsCoding/p/10171201.html推导具体公式
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        # 根据两次sort可以找出元素在升序或者降序中的位置,参考https://blog.csdn.net/LXX516/article/details/78804884
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]

        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses
        N = num_pos.data.sum().double()
        loss_c = loss_c.double()
        loss_four_corners = loss_four_corners.double()
        loss_border = 16.0 * loss_border.double()
        loss_c /= N
        loss_four_corners /= N
        loss_border /= N
        return loss_c, loss_four_corners, loss_border
