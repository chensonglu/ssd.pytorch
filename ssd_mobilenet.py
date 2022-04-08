import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import mobilenet_carplate, change_cfg_for_ssd512_mobilenet
import os
from mobilenet import MobileNetV1


class SSD_mobilenet(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base_net, extras, head, num_classes, head_type):
        super(SSD_mobilenet, self).__init__()
        self.head_type = head_type
        if self.head_type not in ['BB', 'BB+FV', 'FV']:
            assert Exception("head type must be BB, BB+FV, or FV")
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = mobilenet_carplate
        if size == 512:
            self.cfg = change_cfg_for_ssd512_mobilenet(self.cfg)
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.size = size

        # SSD network
        self.base_net = base_net
        self.extras = extras

        self.conf = nn.ModuleList(head[0])
        if self.head_type == 'BB':
            self.loc = nn.ModuleList(head[1])
        elif self.head_type == 'BB+FV':
            self.loc = nn.ModuleList(head[1])
            self.four_corners = nn.ModuleList(head[2])
        else:
            self.four_corners = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            if self.head_type == 'BB':
                self.detect = Detect()
            elif self.head_type == 'BB+FV':
                self.detect = Detect_four_corners()
            else:
                self.detect = Detect_only_four_corners()
            

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        conf = list()
        if self.head_type == 'BB':
            loc = list()
        elif self.head_type == 'BB+FV':
            loc = list()
            four_corners = list()
        else:
            four_corners = list()

        # conv-bw10
        for k in range(11):
            x = self.base_net[k](x)
        sources.append(x)

        # conv-bw12
        for k in range(11, len(self.base_net)):
            x = self.base_net[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            sources.append(x)

        # apply multibox head to source layers
        if self.head_type == 'BB':
            for (x, l, c) in zip(sources, self.loc, self.conf):
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        elif self.head_type == 'BB+FV':
            for (x, l, c, f) in zip(sources, self.loc, self.conf, self.four_corners):
                loc.append(l(x).permute(0, 2, 3, 1).contiguous())
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
                four_corners.append(f(x).permute(0, 2, 3, 1).contiguous())
        else:
            for (x, c, f) in zip(sources, self.conf, self.four_corners):
                conf.append(c(x).permute(0, 2, 3, 1).contiguous())
                four_corners.append(f(x).permute(0, 2, 3, 1).contiguous())

        if self.head_type == 'BB':
            loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        elif self.head_type == 'BB+FV':
            loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
            four_corners = torch.cat([o.view(o.size(0), -1) for o in four_corners], 1)
        else:
            conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
            four_corners = torch.cat([o.view(o.size(0), -1) for o in four_corners], 1)
        if self.phase == "test":
            if self.head_type == 'BB':
                output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                    loc.view(loc.size(0), -1, 4),                   # loc preds
                    self.softmax(conf.view(conf.size(0), -1,
                                self.num_classes)),                # conf preds
                    self.priors.type(type(x.data))                  # default boxes
                )
            elif self.head_type == 'BB+FV':
                output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                    loc.view(loc.size(0), -1, 4),                   # loc preds
                    self.softmax(conf.view(conf.size(0), -1,
                                self.num_classes)),                # conf preds
                    self.priors.type(type(x.data)),                 # default boxes
                    four_corners.view(four_corners.size(0), -1, 8)
                )
            else:
                output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                    self.softmax(conf.view(conf.size(0), -1,
                                self.num_classes)),                # conf preds
                    self.priors.type(type(x.data)),                 # default boxes
                    four_corners.view(four_corners.size(0), -1, 8)
                )
        else:
            if self.head_type == 'BB':
                output = (
                    loc.view(loc.size(0), -1, 4),
                    conf.view(conf.size(0), -1, self.num_classes),
                    self.priors
                )
            elif self.head_type == 'BB+FV':
                output = (
                    loc.view(loc.size(0), -1, 4),
                    conf.view(conf.size(0), -1, self.num_classes),
                    self.priors,
                    four_corners.view(four_corners.size(0), -1, 8)
                )
            else:
                output = (
                    conf.view(conf.size(0), -1, self.num_classes),
                    self.priors,
                    four_corners.view(four_corners.size(0), -1, 8)
                )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def multibox(base_net, extra_layers, cfg, num_classes, head_type='BB'):
    conf_layers = []
    if head_type == 'BB':
        loc_layers = []
    elif head_type == 'BB+FV':
        loc_layers = []
        four_corners_layers = []
    else:
        four_corners_layers = []
    base_net_source = [-3, -1]
    extras_source = [0, 1, 2, 3]
    for k, v in enumerate(base_net_source):
        conf_layers += [nn.Conv2d(base_net[v][3].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
        if head_type == 'BB':
            loc_layers += [nn.Conv2d(base_net[v][3].out_channels,
                        cfg[k] * 4, kernel_size=3, padding=1)]
        elif head_type == 'BB+FV':
            loc_layers += [nn.Conv2d(base_net[v][3].out_channels,
                        cfg[k] * 4, kernel_size=3, padding=1)]
            four_corners_layers += [nn.Conv2d(base_net[v][3].out_channels,
                        cfg[k] * 8, kernel_size=3, padding=1)]
        else:
            four_corners_layers += [nn.Conv2d(base_net[v][3].out_channels,
                        cfg[k] * 8, kernel_size=3, padding=1)]
    for k, v in enumerate(extras_source, 2):
        conf_layers += [nn.Conv2d(extra_layers[v][2].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
        if head_type == 'BB':
            loc_layers += [nn.Conv2d(extra_layers[v][2].out_channels,
                        cfg[k] * 4, kernel_size=3, padding=1)]
        elif head_type == 'BB+FV':
            loc_layers += [nn.Conv2d(extra_layers[v][2].out_channels,
                       cfg[k] * 4, kernel_size=3, padding=1)]
            four_corners_layers += [nn.Conv2d(extra_layers[v][2].out_channels,
                        cfg[k] * 8, kernel_size=3, padding=1)]
        else:
            four_corners_layers += [nn.Conv2d(extra_layers[v][2].out_channels,
                        cfg[k] * 8, kernel_size=3, padding=1)]
    if head_type == 'BB':
        return base_net, extra_layers, (conf_layers, loc_layers)
    elif head_type == 'BB+FV':
        return base_net, extra_layers, (conf_layers, loc_layers, four_corners_layers)
    else:
        return base_net, extra_layers, (conf_layers, four_corners_layers)

mbox = {
    '300': [6, 6, 6, 6, 6, 6],
    '512': [6, 6, 6, 6, 6, 6],
}


def build_ssd(phase, size=300, num_classes=21, head_type='BB'):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300 and size != 512:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 SSD512 (size=300 or size=512) is supported!")
        return
    
    base_net = MobileNetV1(1001).model
    extras = MobileNetV1(1001).extras
    base_, extras_, head_ = multibox(base_net, extras,
                                     mbox[str(size)], num_classes, head_type=head_type)
    return SSD_mobilenet(phase, size, base_, extras_, head_, num_classes, head_type)
