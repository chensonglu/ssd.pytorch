import torch
from torch.autograd import Variable
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.append(".")
from data import CHARS, BaseTransform
from ssd_only_four_corners_recognition import build_ssd
import argparse
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Testing With Pytorch')
parser.add_argument('--input_size', default=300, type=int, help='SSD300 or SSD512')
parser.add_argument('--trained_model',
                    default='weights/voc_weights/VOC300.pth', type=str,
                    help='Trained state_dict file path to open')
args = parser.parse_args()


def draw_box_string(img, position, string):
    """
    img: read by cv;
    string: what you want to draw in img;
    return: img
    """
    x, y = position
    h, w, c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    # simhei.ttf 是字体，你如果没有字体，需要下载
    # CCPD尺寸为1160*720，字号为40，其他数据集以此为基准调整字号
    font = ImageFont.truetype("test/simsun.ttc", int((h*w)/(1160*720)*40), encoding="utf-8")
    draw.text((x, y), string, (0, 0, 255), font=font)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


net = build_ssd('test', args.input_size, 2)    # initialize SSD
net.load_state_dict(torch.load(args.trained_model))
net.eval()
dataset_mean = (104, 117, 123)

# matplotlib inline
from matplotlib import pyplot as plt
from data import CARPLATE_FOUR_CORNERS_RECOGNITIONDetection, CARPLATE_FOUR_CORNERS_RECOGNITIONAnnotationTransform, CARPLATE_FOUR_CORNERS_RECOGNITION_ROOT
testset = CARPLATE_FOUR_CORNERS_RECOGNITIONDetection(CARPLATE_FOUR_CORNERS_RECOGNITION_ROOT, None, transform=BaseTransform(args.input_size, dataset_mean), target_transform=CARPLATE_FOUR_CORNERS_RECOGNITIONAnnotationTransform(),
                                                    dataset_name='test')
for img_id in range(0, 8000, 100):
    image, gt, h, w, labels, lengths = testset.pull_item(img_id)
    rgb_image, img_name = testset.pull_image(img_id)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    xx = Variable(image.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()

    det_out, rec_out = net(xx, [])
    # print('det_out:', det_out.shape)
    # print('rec_out:', rec_out.shape)

    from data import CARPLATE_FOUR_CORNERS_RECOGNITION_CLASSES as labels

    import os
    envpath = '/usr/local/lib/python3.8/dist-packages/cv2/qt/plugins/platforms'
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

    fig = plt.figure(figsize=(10, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    # plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    # [num, num_classes, top_k, 13]
    # 13: score(1) bbox(4) corners(8)
    detections = det_out.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    scale_corner = torch.Tensor(rgb_image.shape[1::-1]).repeat(4)

    # greedy decode
    prebs = rec_out.cpu().detach().numpy()
    # print(prebs)
    preb_labels = list()
    for m in range(prebs.shape[0]):
        preb = prebs[m, :, :]
        preb_label = list()
        for n in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, n], axis=0))
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
    for m, label in enumerate(preb_labels):
        l = ''
        for n in range(len(label)):
            l = l + CHARS[int(label[n])]
        # print(l)

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
            # currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            # currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            rgb_image = draw_box_string(rgb_image, (pt[0], pt[3]), l)

            four_corners = (detections[0, i, j, 5:] * scale_corner).cpu().numpy()
            corners_x = np.append(four_corners[0::2], four_corners[0])
            corners_y = np.append(four_corners[1::2], four_corners[1])
            currentAxis.plot(corners_x, corners_y, linewidth=2, color='red')

            j += 1
    plt.imshow(rgb_image)
    plt.show()
