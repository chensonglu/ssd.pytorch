import torch
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

import sys
sys.path.append(".")

from ssd_offset import build_ssd
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
from data import CAR_CARPLATE_OFFSETDetection, CAR_CARPLATE_OFFSETAnnotationTransform, CAR_CARPLATE_OFFSET_ROOT
testset = CAR_CARPLATE_OFFSETDetection(CAR_CARPLATE_OFFSET_ROOT, None, None, CAR_CARPLATE_OFFSETAnnotationTransform(),
                                       dataset_name='test')
for img_id in range(100):
    image = testset.pull_image(img_id)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = cv2.resize(image, (args.input_size, args.input_size)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()

    y = net(xx)

    from data import CAR_CARPLATE_CLASSES as labels

    fig = plt.figure(figsize=(10,10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    # [num, num_classes, top_k, 10]
    # 10: score(1) bbox(4) has_lp(1) size_lp(2) offset(2)
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)

    for i in range(detections.size(1)):
        # skip background
        if i == 0:
            continue
        j = 0
        th = 0.5
        while detections[0, i, j, 0] > th:
            score = detections[0, i, j, 0]
            has_lp = detections[0, i, j, 5]
            label_name = labels[i-1]
            display_txt = '%.2f' % has_lp
            pt = (detections[0, i, j, 1:5]*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            color = colors[0]
            color_car_center = colors[1]
            color_offset = colors[2]
            color_lp = colors[3]
            color_lp_center = colors[4]
            color_has_lp = colors[5]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color_has_lp, 'alpha':0.5})

            if has_lp > th:
                size_lp_offset = (detections[0, i, j, 6:] * scale).cpu().numpy()
                size_lp = size_lp_offset[:2]
                size_lp = abs(size_lp)
                offset = size_lp_offset[2:]
                center = ((pt[0] + pt[2]) / 2, (pt[1] + pt[3]) / 2)
                center_lp = center + offset
                currentAxis.plot(center[0], center[1], 'o', color = color_car_center, markersize=10)
                currentAxis.plot(center_lp[0], center_lp[1], 'o', color = color_lp_center, markersize=10)
                currentAxis.plot((center[0], center_lp[0]), (center[1], center_lp[1]), color = color_offset)

                coords_lp = center_lp - size_lp / 2, size_lp[0], size_lp[1]
                currentAxis.add_patch(plt.Rectangle(*coords_lp, fill=False, edgecolor=color_lp, linewidth=2))

            j += 1
    plt.show()
