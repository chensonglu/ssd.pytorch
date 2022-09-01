from time import sleep
import torch.nn as nn
import torch



class CNNEncoder_baseline(nn.Module):
	def __init__(self, nc=1):
		super(CNNEncoder_baseline, self).__init__()


		ks = [3, 3, 3, 3, 3, 3, 3, 3, 3]
		ps = [1, 1, 1, 1, 1, 1, 1, 1, 1] 
		ss = [1, 1, 1, 1, 1, 1, 1, 1, 1]
		nm = [32, 32, 32, 64, 64, 64, 128, 128, 128]

		self.conv1 = self.conv_bn_relu(nc, nm[0])
		self.conv2 = self.conv_bn_relu(nm[0], nm[1])
		self.conv3 = self.conv_bn_relu(nm[1], nm[2])
		self.conv4 = self.conv_bn_relu(nm[2], nm[3])
		self.conv5 = self.conv_bn_relu(nm[3], nm[4])
		self.conv6 = self.conv_bn_relu(nm[4], nm[5])
		self.conv7 = self.conv_bn_relu(nm[5], nm[6])
		self.conv8 = self.conv_bn_relu(nm[6], nm[7])
		self.conv9 = self.conv_bn_relu(nm[7], nm[8])

		self.pooling = nn.MaxPool2d(2, 2)


	def conv_bn_relu(self,c_in,c_out,kernel=3,stride=1,pad=1):
		layers = nn.Sequential(
			nn.Conv2d(c_in, c_out, kernel, stride, pad),
			nn.BatchNorm2d(c_out),
			nn.ReLU(True))
		return nn.Sequential(*layers)

	def forward(self, input, tsne=False):
		decouple_outs = []
		text_features = []
		out1 = self.conv1(input)
		out2 = self.conv2(out1)
		out3 = self.conv3(out2)
		out4 = self.conv4(self.pooling(out3))
		
		out5 = self.conv5(out4)
		out6 = self.conv6(out5)
		out7 = self.conv7(self.pooling(out6))
		out8 = self.conv8(out7)
		out9 = self.conv9(out8)
		text_features.append(out3)

		conv_out = out9

		return conv_out

