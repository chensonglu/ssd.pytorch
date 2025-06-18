import numpy as np
import sys, os
import time
import cv2
sys.path.append(os.getcwd())
# crnn packages
import torch
from torch.autograd import Variable
import model_crnn_fc as crnn
import Levenshtein

import argparse
parser = argparse.ArgumentParser()
# image annotatin file 
parser.add_argument('--images_path', type=str, default='/data/CCPD/result/paper/CCPD_gt.txt', help='the path to your images')
opt = parser.parse_args()


# crnn params
crnn_model_path = './crnn_best.pth'
alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新领学使警挂港澳电'
nclass = len(alphabet)+1


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()

        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

# crnn文本信息识别
def crnn_recognition(cropped_image, model):
    """Decode encoded texts back into strs.

        Args:
            numpy: a RGB license plate image
            model: <class 'models.crnn_fc.CRNN_FC'>

        Returns:
            str: predict result
    """

    imgH = 32
    imgW = 100
    mean = 0.588
    std = 0.193
    converter = strLabelConverter(alphabet)
    # print(alphabet)
    image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    ### ratio
    ### 280是中文训练集中图片的宽度，160是将原始图片缩小后的图片宽度
    w_now = int(image.shape[1] / (100 * 1.0 / imgW))
    h, w = image.shape
    image = cv2.resize(image, (0,0), fx=imgW/w, fy=imgH/h, interpolation=cv2.INTER_CUBIC)

    image = (np.reshape(image, (imgH, imgW, 1))).transpose(2, 0, 1)

    image = image.astype(np.float32) / 255.
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image.sub_(mean).div_(std)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)

    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    # print('results: {0}'.format(sim_pred))

    return sim_pred

if __name__ == '__main__':

	# crnn_FC network
    ## load model
    model = crnn.CRNN_FC(32, 1, nclass, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from {0}'.format(crnn_model_path))
   
    model.load_state_dict(torch.load(crnn_model_path)) 
    
    started = time.time()
    
    ## count the number of corrects 
    count_all = 0           ## the number of all samples
    count_right = 0         ##the number of corrects 
    count_Chinese = 0
    count_woChinese = 0     ##the number of corrects without chinese
    count_ed1 = 0           ## the number of samples: edit distance <= 1


    image_root = '/data/CCPD/result/paper/FV_CIoU'

    with open(opt.images_path, 'r' ,encoding='utf-8') as f:
        for item in f.readlines():
            count_all += 1
            if count_all % 1000 == 0:
                print(count_all)
            image_path = os.path.join(image_root,item.strip().split(' ')[0])
            label = item.strip().split(' ')[1]
            image = cv2.imread(image_path)  # load license plate image
            predict = crnn_recognition(image, model)  # the predict result
            print(item.strip().split(' ')[0] + ' ' + predict)
            lev_distance = Levenshtein.distance(label,predict)
            if lev_distance == 0:
                count_right += 1
            if label[0]==predict[0]:
                count_Chinese += 1
            if label[1:]==predict[1:]:
                count_woChinese += 1
            if lev_distance <= 1:
                count_ed1 +=1


    finished = time.time()
    print('elapsed time: {0}'.format(finished-started))
    
    print('accuracy:   ', str(count_right), str(count_all), str(1.0*count_right/count_all))
    print('accuracy_ed1:   ', str(count_ed1), str(count_all), str(1.0*count_ed1/count_all))
    print('accuracy_woChinese:   ', str(count_woChinese), str(count_all), str(1.0*count_woChinese/count_all))
    print('accuracy_Chinese:   ', str(count_Chinese), str(count_all), str(1.0*count_Chinese/count_all))
