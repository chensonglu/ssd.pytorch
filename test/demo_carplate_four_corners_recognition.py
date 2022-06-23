import torch
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

import sys
sys.path.append(".")

from data import CHARS

from ssd_only_four_corners_recognition import build_ssd
import argparse

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Testing With Pytorch')
parser.add_argument('--input_size', default=300, type=int, help='SSD300 or SSD512')
parser.add_argument('--trained_model',
                    default='weights/voc_weights/VOC300.pth', type=str,
                    help='Trained state_dict file path to open')
args = parser.parse_args()

net = build_ssd('test', args.input_size, 2)    # initialize SSD
net.load_weights(args.trained_model)

# matplotlib inline
from matplotlib import pyplot as plt
from data import CARPLATE_FOUR_CORNERS_RECOGNITIONDetection, CARPLATE_FOUR_CORNERS_RECOGNITIONAnnotationTransform, CARPLATE_FOUR_CORNERS_RECOGNITION_ROOT
testset = CARPLATE_FOUR_CORNERS_RECOGNITIONDetection(CARPLATE_FOUR_CORNERS_RECOGNITION_ROOT, None, None, CARPLATE_FOUR_CORNERS_RECOGNITIONAnnotationTransform(),
                                         dataset_name='test')
for img_id in range(5):
    image, _ = testset.pull_image(img_id)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = cv2.resize(image, (args.input_size, args.input_size)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()

    det_out, rec_out = net(xx, [])
    print('det_out:', det_out.shape)
    print('rec_out:', rec_out.shape)

    from data import CARPLATE_FOUR_CORNERS_RECOGNITION_CLASSES as labels

    fig = plt.figure(figsize=(10, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    # [num, num_classes, top_k, 13]
    # 13: score(1) bbox(4) corners(8)
    detections = det_out.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    scale_corner = torch.Tensor(rgb_image.shape[1::-1]).repeat(4)

    # greedy decode
    rec_out = rec_out.cpu().detach().numpy()
    preb_labels = list()
    for i in range(rec_out.shape[0]):
        preb = rec_out[i, :, :]
        preb_label = list()
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label: # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)
    print('preb_labels:', preb_labels)
    for i, label in enumerate(preb_labels):
        lb = ""
        for i in label:
            lb += CHARS[i]
        print('lb:', lb)

    for i in range(detections.size(1)):
        # skip background
        if i == 0:
            continue
        j = 0
        th = 0.5
        while detections[0, i, j, 0] > th:
            score = detections[0, i, j, 0]
            label_name = labels[i-1]
            display_txt = '%s: %.2f' % (label_name, score)
            pt = (detections[0, i, j, 1:5] * scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})

            four_corners = (detections[0, i, j, 5:] * scale_corner).cpu().numpy()
            corners_x = np.append(four_corners[0::2], four_corners[0])
            corners_y = np.append(four_corners[1::2], four_corners[1])
            currentAxis.plot(corners_x, corners_y, linewidth=3)

            j += 1

    plt.show()
