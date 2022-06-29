"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import sys
sys.path.append(".")

from data import *
import torch.utils.data as data

import sys
import os
import time
import numpy as np
import pickle
import cv2
from log import log
from evaluation.mean_average_precision import MetricBuilder

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

args_cuda = True


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if torch.cuda.is_available():
    if args_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args_cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


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


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = 0
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_12=False):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_12_metric = use_12
    print('VOC12 metric? ' + ('Yes' if use_12_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        metrics, map = voc_eval(
           filename, annopath, imgsetpath.format(set_type), cls, cachedir,
           ovthresh=0.75, use_12_metric=use_12_metric)
        aps += [map]
        print('AP for {} = {:.4f}'.format(cls, map))
        # with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        #     pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        log.l.info('AP for {} = {:.4f}'.format(cls, map))
    log.l.info('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.4f}'.format(ap))
    print('{:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('refer to https://github.com/bes-dev/mean_average_precision')
    print('--------------------------------------------------------------')


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.75,
             use_12_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [use_12_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.75)
[use_12_metric]: Whether to use VOC12's all points AP computation
   (default True)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # create metric_fn
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:
        # exchange dets as key-value dic
        preds_dic = {}
        splitlines = [x.strip().split(' ') for x in lines]
        for x in splitlines:
            if x[0] in preds_dic.keys():
                preds_dic[x[0]] = np.vstack((preds_dic[x[0]], np.append(np.array([float(z) for z in x[2:]]), [int(0), float(x[1])])))
            else:
                preds_dic[x[0]] = np.append(np.array([float(z) for z in x[2:]]), [int(0), float(x[1])])
                preds_dic[x[0]] = np.expand_dims(preds_dic[x[0]], axis=0)

        # extract gt objects for this class
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            GT_bbox = np.array([x['bbox'] for x in R])
            GT_append = np.zeros((GT_bbox.shape[0], 3), dtype=np.int)
            GTs = np.hstack((GT_bbox, GT_append))
            if imagename not in preds_dic.keys():
                preds = np.array([[]])
            else:
                preds = preds_dic[imagename]
            metric_fn.add(preds, GTs)

        metrics = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')
        map = metrics["mAP"]
        # metrics = metric_fn.value(iou_thresholds=ovthresh)
        # ap = metrics[ovthresh][0]['ap']
        # rec = metrics[ovthresh][0]['recall']
        # prec = metrics[ovthresh][0]['precision']
    else:
        metrics = {}
        map = -1.

    return metrics, map


def test_net(save_folder, obj_type, voc_root, set_type_, labelmap_, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    global annopath
    global imgpath
    global imgsetpath
    global devkit_path
    global set_type
    global labelmap
    annopath = os.path.join(voc_root, 'Annotations', '%s.xml')
    imgpath = os.path.join(voc_root, 'JPEGImages', '%s.jpg')
    imgsetpath = os.path.join(voc_root, 'ImageSets', 'Main', '{:s}.txt')
    devkit_path = voc_root
    set_type = set_type_
    labelmap = labelmap_

    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir(save_folder + '/ssd' + str(im_size) + '_' + obj_type, set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    total_time = 0
    for i in range(num_images):
        if obj_type in ['carplate_only_four_corners_with_border_recognition']:
            im, gt, h, w, labels, lenghts = dataset.pull_item(i)
        else:
            im, gt, h, w = dataset.pull_item(i)
        x = Variable(im.unsqueeze(0))
        if args_cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        if obj_type in ['two_stage_end2end', 'two_stage_end2end_with_border', 'TITS_Neuro']:
            detections = net(x, []).data
        elif obj_type in ['carplate_only_four_corners_with_border_recognition']:
            det_out, rec_out = net(x, [])
            detections = det_out.data
        else:
            detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            if obj_type in ['VOC', 'COCO', 'car', 'carplate', 'car_carplate', 'two_branch']:
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
            elif obj_type == 'car_carplate_offset':
                mask = dets[:, 0].gt(0.).expand(10, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 10)
            elif obj_type in ['carplate_four_corners','carplate_four_corners_with_border',
                            'carplate_only_four_corners', 'carplate_only_four_corners_with_border',
                            'two_stage_end2end', 'two_stage_end2end_with_border',
                            'TITS_Neuro', 'carplate_only_four_corners_with_border_recognition']:
                mask = dets[:, 0].gt(0.).expand(13, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 13)
            else:
                assert False, "wrong object type"
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:5]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))
        if i > 0:
            total_time += detect_time
    print("average time: " + str(total_time / (num_images-1) * 1000) + ' ms')

    # with open(det_file, 'wb') as f:
    #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)
