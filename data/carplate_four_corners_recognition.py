import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from .config import *
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

CARPLATE_FOUR_CORNERS_RECOGNITION_CLASSES = (  # always index 0
    'carplate', )

# note: if you used our download scripts, this should be right
CARPLATE_FOUR_CORNERS_RECOGNITION_ROOT = osp.join('/home/yzbj/data', 'CLPD/VOC/')


class CARPLATE_FOUR_CORNERS_RECOGNITIONAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False, object_size='all'):
        self.class_to_ind = class_to_ind or dict(
            zip(CARPLATE_FOUR_CORNERS_RECOGNITION_CLASSES, range(len(CARPLATE_FOUR_CORNERS_RECOGNITION_CLASSES))))
        self.keep_difficult = keep_difficult
        self.object_size = object_size

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            x_top_left = int(bbox.find('x_top_left').text) - 1
            y_top_left = int(bbox.find('y_top_left').text) - 1
            x_bottom_left = int(bbox.find('x_bottom_left').text) - 1
            y_bottom_left = int(bbox.find('y_bottom_left').text) - 1
            x_bottom_right = int(bbox.find('x_bottom_right').text) - 1
            y_bottom_right = int(bbox.find('y_bottom_right').text) - 1

            QP = np.array([x_top_left - x_bottom_left, y_top_left - y_bottom_left])
            v = np.array([x_bottom_left - x_bottom_right, y_bottom_left - y_bottom_right])
            h = np.linalg.norm(np.cross(QP, v))/np.linalg.norm(v)
            
            if self.object_size != 'all':
                if (h <= 16 and self.object_size != 'small') or (h > 16 and h <= 32 and self.object_size != 'medium') or (h > 32 and self.object_size != 'large'):
                    continue

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            # four corners
            corners = ['x_top_left', 'y_top_left', 'x_top_right', 'y_top_right', 'x_bottom_right', 'y_bottom_right',
                       'x_bottom_left', 'y_bottom_left']
            for i, corner in enumerate(corners):
                cur_corner = int(bbox.find(corner).text) - 1
                # scale height or width
                cur_corner = cur_corner / width if i % 2 == 0 else cur_corner / height
                bndbox.append(cur_corner)
            # label -1
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            # LP number
            bndbox.append(obj.find('number').text)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, x_top_left, y_top_left, ..., label_ind, number]
            # img_id = target.find('filename').text[:-4]
        return res  # [[xmin, ymin, xmax, ymax, x_top_left, y_top_left, ..., label_ind, number], ... ]


class CARPLATE_FOUR_CORNERS_RECOGNITIONDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=None,
                 transform=None, target_transform=CARPLATE_FOUR_CORNERS_RECOGNITIONAnnotationTransform(keep_difficult=True, object_size='all'),
                 dataset_name='trainval'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for line in open(osp.join(self.root, 'ImageSets', 'Main', self.name + '.txt')):
            self.ids.append((self.root, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w, label = self.pull_item(index)
        return im, gt, label

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        # # 这种方式是按照CCPD名字得到label
        # imgname = ''
        # basename = img_id[1]
        # numbers = basename.strip().split('-')[-3]
        # province = provinces[int(numbers.split('_')[0])]
        # imgname = imgname + province
        # alphabet = alphabets[int(numbers.split('_')[1])]
        # imgname = imgname + alphabet
        # for i in range(5):
        #     ad = ads[int(numbers.split('_')[i+2])]
        #     imgname = imgname + ad

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        numbers = [t[-1] for t in target]
        number_ids = []
        for number in numbers:
            number_id = []
            for c in number:
                try:
                    number_id.append(CHARS_DICT[c]+1)
                except Exception as e:
                    pass # CLPD中有"警",暴力跳过
            for i in range(K-len(number_id)):
                number_id.append(0)
            number_ids += [number_id]
            
        target = [t[:-1] for t in target]
        if self.transform is not None:
            target = np.array(target)
            if target.shape[0] > 0:
                img, boxes, labels = self.transform(img, target[:, :-1], target[:, -1])
                # to rgb
                img = img[:, :, (2, 1, 0)]
                # img = img.transpose(2, 0, 1)
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width, number_ids
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR), img_id[1]

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
