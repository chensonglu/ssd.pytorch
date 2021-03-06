import torch
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

import sys
sys.path.append(".")

from ssd_TITS_Neuro import build_ssd
import argparse

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Testing With Pytorch')
parser.add_argument('--input_size', default=300, type=int, help='SSD300 or SSD512')
parser.add_argument('--input_size_2', default=56, type=int, help='input size of the second network')
parser.add_argument('--expand_num', default=3, type=int, help='expand ratio around the license plate')
parser.add_argument('--trained_model',
                    default='weights/voc_weights/VOC300.pth', type=str,
                    help='Trained state_dict file path to open')
args = parser.parse_args()

net = build_ssd('test', args.input_size, args.input_size_2, 2, args.expand_num)    # initialize SSD
net.load_weights(args.trained_model)

# matplotlib inline
from matplotlib import pyplot as plt
from data import CAR_CARPLATE_TWO_STAGE_END2ENDDetection, CAR_CARPLATE_TWO_STAGE_END2ENDAnnotationTransform, CAR_CARPLATE_TWO_STAGE_END2END_ROOT
testset = CAR_CARPLATE_TWO_STAGE_END2ENDDetection(CAR_CARPLATE_TWO_STAGE_END2END_ROOT, None, None, CAR_CARPLATE_TWO_STAGE_END2ENDAnnotationTransform(),
                                       dataset_name='test')
for img_id in range(2000):
    image = testset.pull_image(img_id)
    # if image is None:
    #     continue
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = cv2.resize(image, (args.input_size, args.input_size)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()

    # 给空的targets,车辆检测结果作为targets
    # 在内部forward的过程中已经进行了阈值过滤,跟后续展示代码相同
    detections = net(xx, [])

    from data import CAR_CARPLATE_CLASSES as labels

    fig = plt.figure(figsize=(10, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    # [num, num_classes, num_car, 10]
    # 10: score(1) bbox(4) has_lp(1) size_lp(2) offset(2)
    detections = detections.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    scale_4 = torch.Tensor(rgb_image.shape[1::-1]).repeat(4)

    for i in range(detections.size(1)):
        # skip background
        if i == 0:
            continue
        th = 0.5
        for j in range(detections.size(2)):
            if detections[0, i, j, 0] > th:
                label_name = labels[i-1]
                display_txt = '%s: %.2f' % (label_name, detections[0, i, j, 0])
                pt = (detections[0, i, j, 1:5]*scale).cpu().numpy()
                coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                color = colors[i]
                if i == 1:
                    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='#ffc000', linewidth=2))
                    # currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
                
                if i == 2:
                    lp_pt = (detections[0, i, j, 1:5]*scale).cpu().numpy()
                    lp_coords = (lp_pt[0], lp_pt[1]), lp_pt[2] - lp_pt[0] + 1, lp_pt[3] - lp_pt[1] + 1
                    # currentAxis.add_patch(plt.Rectangle(*lp_coords, fill=False, edgecolor=colors[1], linewidth=2))
                    four_corners = (detections[0, i, j, 5:]*scale_4).cpu().numpy()
                    corners_x = np.append(four_corners[0::2], four_corners[0])
                    corners_y = np.append(four_corners[1::2], four_corners[1])
                    currentAxis.plot(corners_x, corners_y, linewidth=2, color='#ff0000')
    plt.show()





    # car_width = pt[2] - pt[0] + 1
    # car_height = pt[3] - pt[1] + 1
    # # 让所有车都在图片范围内
    # detections[:, :, :, 1:5] = torch.max(detections[:, :, :, 1:5], torch.zeros(detections[:, :, :, 1:5].shape))
    # detections[:, :, :, 1:5] = torch.min(detections[:, :, :, 1:5], torch.ones(detections[:, :, :, 1:5].shape))
    # for i in range(detections.size(1)):
    #     # skip background
    #     if i == 0:
    #         continue
    #     th = 0.6
    #     for j in range(detections.size(2)):
    #         score = detections[0, i, j, 0]
    #         label_name = labels[i - 1]
    #         display_txt = '%s: %.2f' % (label_name, score)
    #         pt = (detections[0, i, j, 1:5] * scale).cpu().numpy()
    #         coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
    #         currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=colors[0], linewidth=2))
    #         # currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
    #         car_width = pt[2] - pt[0] + 1
    #         car_height = pt[3] - pt[1] + 1
    #         for k in range(1):
    #             if detections_2[j, 1, k, 0] > th:
    #                 lp_pt = (detections_2[j, 1, k, 1:5] * torch.Tensor([car_width, car_height]).repeat(2)).cpu().numpy()
    #                 lp_coords = (pt[0] + lp_pt[0], pt[1] + lp_pt[1]), lp_pt[2] - lp_pt[0] + 1, lp_pt[3] - lp_pt[1] + 1
    #                 currentAxis.add_patch(plt.Rectangle(*lp_coords, fill=False, edgecolor=colors[1], linewidth=2))
    #
    # plt.show()
