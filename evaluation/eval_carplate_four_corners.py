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

from data import CARPLATE_FOUR_CORNERSDetection, CARPLATE_FOUR_CORNERSAnnotationTransform, CARPLATE_FOUR_CORNERS_ROOT, BaseTransform
from data import CARPLATE_FOUR_CORNERS_CLASSES as labelmap
import torch.utils.data as data
import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
import shared_function as sf
from log import log

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=CARPLATE_FOUR_CORNERS_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--input_size', default=300, type=int,
                    help='SSD300 OR SSD512')
parser.add_argument('--iou_thres', default='0.5,0.75', type=str,
                    help='VOC2012 IoU threshold, separated by ",", such as 0.5,0.75')
parser.add_argument('--obj_type', default='carplate_four_corners', choices=['carplate', 'carplate_four_corners', 'carplate_only_four_corners'],
                    type=str, help='Object type')
parser.add_argument('--object_size', default='all', type=str,
                    help='Size to test: all, small, medium or large')

args = parser.parse_args()

bbox_cal_types = None
four_corners_cal_types = None
if args.obj_type == "carplate":
    from ssd import build_ssd
    print("carplate")
    bbox_cal_types = ['bbox']
    four_corners_cal_types = ['fc_from_bbox']
if args.obj_type == "carplate_four_corners":
    from ssd_four_corners import build_ssd
    print("carplate_four_corners")
    bbox_cal_types = ['bbox']
    four_corners_cal_types = ['fc']
elif args.obj_type == "carplate_only_four_corners":
    from ssd_only_four_corners import build_ssd
    print("carplate_only_four_corners")
    bbox_cal_types = ['bbox_from_fc']
    four_corners_cal_types = ['fc']

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.voc_root, 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'ImageSets',
                          'Main', '{:s}.txt')
devkit_path = args.voc_root
dataset_mean = (104, 117, 123)
set_type = 'test'


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


def parse_rec(filename, object_size):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        # difficult always 0
        obj_struct['difficult'] = 0
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        obj_struct['four_corners'] = [int(bbox.find('x_top_left').text) - 1,
                              int(bbox.find('y_top_left').text) - 1,
                              int(bbox.find('x_top_right').text) - 1,
                              int(bbox.find('y_top_right').text) - 1,
                              int(bbox.find('x_bottom_right').text) - 1,
                              int(bbox.find('y_bottom_right').text) - 1,
                              int(bbox.find('x_bottom_left').text) - 1,
                              int(bbox.find('y_bottom_left').text) - 1]

        x_top_left = int(bbox.find('x_top_left').text) - 1
        y_top_left = int(bbox.find('y_top_left').text) - 1
        x_bottom_left = int(bbox.find('x_bottom_left').text) - 1
        y_bottom_left = int(bbox.find('y_bottom_left').text) - 1
        x_bottom_right = int(bbox.find('x_bottom_right').text) - 1
        y_bottom_right = int(bbox.find('y_bottom_right').text) - 1
        QP = np.array([x_top_left - x_bottom_left, y_top_left - y_bottom_left])
        v = np.array([x_bottom_left - x_bottom_right, y_bottom_left - y_bottom_right])
        h = np.linalg.norm(np.cross(QP, v))/np.linalg.norm(v)
        if object_size != 'all':
            if (h <= 16 and object_size != 'small') or (h > 16 and h <= 32 and object_size != 'medium') or (h > 32 and object_size != 'large'):
                continue

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
                if dets is None or len(dets) == 0:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                            dets[k, 0] + 1, dets[k, 1] + 1,
                            dets[k, 2] + 1, dets[k, 3] + 1,
                            dets[k, 4] + 1, dets[k, 5] + 1,
                            dets[k, 6] + 1, dets[k, 7] + 1,
                            dets[k, 8] + 1, dets[k, 9] + 1,
                            dets[k, 10] + 1, dets[k, 11] + 1))


def do_python_eval(output_dir='output', use_12=False, object_size='all'):
    log.l.info("_".join(args.trained_model.split('/')[1:]) + '-' + args.voc_root.split('/')[-2])
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    # The PASCAL VOC metric changed in 2010
    use_12_metric = use_12
    print('VOC12 metric? ' + ('Yes' if use_12_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for _, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        # AP for horizontal bbox
        if object_size == 'all':
            for cal_type in bbox_cal_types:
                metrics, map = eval_AP(
                    filename, annopath, imgsetpath.format(set_type), cls, cachedir, cal_type,
                    use_12_metric=use_12_metric, object_size=object_size)
                save_name = "_".join(args.trained_model.split('/')[1:]) + '-' + args.voc_root.split('/')[-2] + '-' + cal_type + '-' + 'coco'
                with open(os.path.join(output_dir, cls + '-' + save_name + '-pr.pkl'), 'wb') as f:
                    pickle.dump({'metrics': metrics}, f)
                if cal_type == 'bbox':
                    log.l.info('AP for {} with COCO = {:.4f}'.format(cls, map))
                    print('AP for {} with COCO = {:.4f}'.format(cls,  map))
                elif cal_type == 'bbox_from_fc':
                    log.l.info('AP for {} with COCO from four corners = {:.4f}'.format(cls, map))
                    print('AP for {} with COCO from four corners = {:.4f}'.format(cls, map))
        # F1-score for quadarilateral bbox
        for cal_type in four_corners_cal_types:
            rec_list = []
            prec_list = []
            F1_list = []
            for t in args.iou_thres.split(','):
                rec, prec, F1 = eval_F1_score(
                    filename, annopath, imgsetpath.format(set_type), cls, cachedir, cal_type,
                    ovthresh=float(t.strip()), use_12_metric=use_12_metric, object_size=object_size)
                rec_list.append(rec)
                prec_list.append(prec)
                F1_list.append(F1)
                if cal_type == 'fc':
                    log.l.info('{} with IoU threshold {} => recall: {:.4f}, precision: {:.4f}, F1-score: {:.4f}'.format(
                        cls, t.strip(), rec, prec, F1))
                    print('{} with IoU threshold {} => recall: {:.4f}, precision: {:.4f}, F1-score: {:.4f}'.format(
                        cls, t.strip(), rec, prec, F1))
                elif cal_type == 'fc_from_bbox':
                    log.l.info('{} with IoU threshold {} from bbox => recall: {:.4f}, precision: {:.4f}, F1-score: {:.4f}'.format(
                        cls, t.strip(), rec, prec, F1))
                    print('{} with IoU threshold {} from bbox => recall: {:.4f}, precision: {:.4f}, F1-score: {:.4f}'.format(
                        cls, t.strip(), rec, prec, F1))
            log.l.info('{} with Mean => recall: {:.4f}, precision: {:.4f}, F1-score: {:.4f}'.format(
                cls, np.mean(np.array(rec_list)), np.mean(np.array(prec_list)), np.mean(np.array(F1_list))))
            print('{} with Mean => recall: {:.4f}, precision: {:.4f}, F1-score: {:.4f}'.format(
                cls, np.mean(np.array(rec_list)), np.mean(np.array(prec_list)), np.mean(np.array(F1_list))))
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('refer to https://github.com/bes-dev/mean_average_precision')
    print('--------------------------------------------------------------')


def eval_AP(detpath,
            annopath,
            imagesetfile,
            classname,
            cachedir,
            cal_type,
            ovthresh=0.5,
            use_12_metric=False,
            object_size='all'):
    """
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
cal_type: bbox or bbox from four corners
[ovthresh]: Overlap threshold (default = 0.5)
[use_12_metric]: Whether to use VOC12's all points AP computation
   (default True)
object_size: all, small, medium, large, not finished yet
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
            parse_results = parse_rec(annopath % (imagename), object_size)
            if parse_results == []:
                continue
            recs[imagename] = parse_results
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

    metrics, map = sf.calculate_AP(ovthresh, recs, detpath, classname, imagenames, cal_type=cal_type)

    return metrics, map


def eval_F1_score(detpath,
                  annopath,
                  imagesetfile,
                  classname,
                  cachedir,
                  cal_type,
                  ovthresh=0.5,
                  use_12_metric=True,
                  object_size='all'):
    """
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
cal_type: four corners or four corners from bbox
[ovthresh]: Overlap threshold (default = 0.5)
[use_12_metric]: Whether to use VOC12's all points AP computation
   (default True)
object_size: all, small, medium, large, finished
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
            parse_results = parse_rec(annopath % (imagename), object_size)
            if parse_results == []:
                continue
            recs[imagename] = parse_results
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

    rec, prec, F1 = sf.calculate_F1_score(ovthresh, recs, detpath, classname, cal_type=cal_type)

    return rec, prec, F1


def test_net(save_folder, net, cuda, dataset, im_size=300, object_size='all'):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 13 array of detections in
    #    (bbox(4) four_corners(8) score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir(save_folder + '/ssd_carplate_bbox_or_four_corners', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    total_time = 0
    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)
        if gt.shape[0] < 1:
            continue
        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            if args.obj_type in ['carplate_four_corners', 'carplate_only_four_corners']:
                mask = dets[:, 0].gt(0.).expand(13, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 13)
            elif args.obj_type == 'carplate':
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            if args.obj_type in ['carplate_four_corners', 'carplate_only_four_corners']:
                boxes[:, 4] *= w
                boxes[:, 6] *= w
                boxes[:, 8] *= w
                boxes[:, 10] *= w
                boxes[:, 5] *= h
                boxes[:, 7] *= h
                boxes[:, 9] *= h
                boxes[:, 11] *= h
            elif args.obj_type == 'carplate':
                boxes_append = torch.zeros(boxes.shape[0], 8)
                boxes_append[:, 0] = boxes[:, 0]
                boxes_append[:, 2] = boxes[:, 2]
                boxes_append[:, 4] = boxes[:, 2]
                boxes_append[:, 6] = boxes[:, 0]
                boxes_append[:, 1] = boxes[:, 1]
                boxes_append[:, 3] = boxes[:, 1]
                boxes_append[:, 5] = boxes[:, 3]
                boxes_append[:, 7] = boxes[:, 3]
                boxes = torch.cat((boxes, boxes_append), 1)
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
    print("FPS: " + str(1000.0 / (total_time / (num_images-1) * 1000.0)))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset, object_size)


def evaluate_detections(box_list, output_dir, dataset, object_size):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir, object_size=object_size)


if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1                      # +1 for background
    net = build_ssd('test', args.input_size, num_classes)            # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = CARPLATE_FOUR_CORNERSDetection(root=args.voc_root,
                           transform=BaseTransform(args.input_size, dataset_mean),
                           target_transform=CARPLATE_FOUR_CORNERSAnnotationTransform(keep_difficult=True, object_size=args.object_size),
                           dataset_name=set_type)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset, args.input_size, object_size=args.object_size)
