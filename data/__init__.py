from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT
from .car_carplate import CAR_CARPLATEDetection, CAR_CARPLATEAnnotationTransform, CAR_CARPLATE_CLASSES, CAR_CARPLATE_ROOT
from .car import CARDetection, CARAnnotationTransform, CAR_CLASSES, CAR_ROOT
from .carplate import CARPLATEDetection, CARPLATEAnnotationTransform, CARPLATE_CLASSES, CARPLATE_ROOT
from .car_carplate_offset import CAR_CARPLATE_OFFSETDetection, CAR_CARPLATE_OFFSETAnnotationTransform, CAR_CARPLATE_OFFSET_CLASSES, CAR_CARPLATE_OFFSET_ROOT
from .carplate_four_corners import CARPLATE_FOUR_CORNERSDetection, CARPLATE_FOUR_CORNERSAnnotationTransform, CARPLATE_FOUR_CORNERS_CLASSES, CARPLATE_FOUR_CORNERS_ROOT
from .carplate_four_corners_recognition import CARPLATE_FOUR_CORNERS_RECOGNITIONDetection, CARPLATE_FOUR_CORNERS_RECOGNITIONAnnotationTransform, CARPLATE_FOUR_CORNERS_RECOGNITION_CLASSES, CARPLATE_FOUR_CORNERS_RECOGNITION_ROOT
from .car_carplate_two_stage_end2end import CAR_CARPLATE_TWO_STAGE_END2ENDDetection, CAR_CARPLATE_TWO_STAGE_END2ENDAnnotationTransform, CAR_CARPLATE_TWO_STAGE_END2END_CLASSES, CAR_CARPLATE_TWO_STAGE_END2END_ROOT
from .coco import COCODetection, COCOAnnotationTransform, COCO_CLASSES, COCO_ROOT, get_label_map
from .config import *
import torch
import cv2
import numpy as np

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def detection_recognition_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    labels = []
    lengths = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        labels.extend(sample[2])
        lengths.append(sample[3])
    labels = np.asarray(labels).flatten().astype(np.int)
    return torch.stack(imgs, 0), torch.stack(targets, 0), torch.from_numpy(labels), lengths


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
