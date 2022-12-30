import torch.nn as nn
import torch
import torch.nn.functional as F
from PVAM.encoder import *
from PVAM.decoder import *
from PVAM.Trans_atten import *
from PVAM.RNN_atten import *


class LPR_model_trans_baseline(nn.Module):
	def __init__(self, nc, nclass, imgW=96, imgH=32, K=8, n_position=192, head=4, inner=512, model_dim=256, isSeqModel=True):
		super(LPR_model_trans_baseline, self).__init__()
		self.encoder = CNNEncoder_baseline(nc)
		self.isSeqModel = isSeqModel
		if self.isSeqModel:
			key_dim = int(model_dim/head)
			self.SequenceModeling = Transforme_Encoder(d_word_vec=model_dim, n_layers=2, n_head=head, d_k=key_dim, d_v=key_dim,
				d_model=model_dim, d_inner=inner, dropout=0.1, n_position=n_position)
		self.attention = PP_layer(n_dim=model_dim, N_max_character=8, n_position=n_position)
		self.decoder = FCDecoder(nclass, input_dim=model_dim)
		
	def forward(self, input):
		conv_out = self.encoder(input)
		b, c, h, w = conv_out.shape
		conv_out = conv_out.permute(0, 1, 3, 2)
		conv_out = conv_out.contiguous().view(b, c, -1)
		conv_out = conv_out.permute(0, 2, 1)
		if self.isSeqModel:
			seq_out = self.SequenceModeling(conv_out, src_mask=None)[0]
			atten_out, atten = self.attention(seq_out)
		else:
			atten_out, atten = self.attention(conv_out)
		text_preds = self.decoder(atten_out)
		return atten_out, text_preds


class BidirectionalLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(BidirectionalLSTM, self).__init__()
		self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
		self.linear = nn.Linear(hidden_size * 2, output_size)

	def forward(self, input):
		"""
		input : visual feature [batch_size x T x input_size]
		output : contextual feature [batch_size x T x output_size]
		"""
		self.rnn.flatten_parameters()
		recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
		output = self.linear(recurrent)  # batch_size x T x output_size
		return output


class LPR_model_RNNAtten_baseline(nn.Module):
	"""
        input:
            is1D : The attention is calculated on 1D or 2D visual
			isLSTM : Set True to add two LSTM to build sequence modelling (original RNN attention)
            isSequenceModeling : whether use self attention
        	n_position,head,inner: the params of self attention module
			model_dim: the channel number of input features
	"""
	def __init__(self, nc, nclass, is1D=True, isLSTM=True, n_position=192, head=2, inner=512, isSequenceModeling=True, model_dim=128):
		super(LPR_model_RNNAtten_baseline, self).__init__()
		self.model_dim = model_dim
		self.encoder = CNNEncoder_baseline(nc)
		nclass += 1 # add two token, [GO(0)]/[End(1)], 之前的模型都加了[black(0)] token,所以这里加1
		self.attention = Attention_RNN(input_size=self.model_dim, hidden_size=model_dim, num_classes=nclass)
		self.decoder = FCDecoder(nclass, input_dim=self.model_dim)
		self.is1D = is1D
		self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
		self.isSequenceModeling = isSequenceModeling
		input_dim = self.model_dim
		hidden_dim = self.model_dim
		self.isLSTM = isLSTM
		if self.isLSTM:
			self.SequenceModeling = nn.Sequential(
				BidirectionalLSTM(input_dim, hidden_dim, hidden_dim),
				BidirectionalLSTM(hidden_dim, hidden_dim, input_dim))
		if self.isSequenceModeling:
			key_dim = int(model_dim/head)
			self.SequenceModeling2 = Transforme_Encoder(d_word_vec=model_dim, n_layers=2, n_head=head, d_k=key_dim, d_v=key_dim,
			d_model=model_dim, d_inner=inner, dropout=0.1, n_position=n_position)

	def forward(self, input, text, is_train=True, isVis=False):
		conv_out = self.encoder(input)
		if self.is1D:
			conv_out = self.AdaptiveAvgPool(conv_out.permute(0, 3, 1, 2))
		else:
			conv_out = conv_out.reshape(conv_out.size(0), conv_out.size(1), 1, -1).permute(0, 3, 1, 2)
		conv_out = conv_out.squeeze(3)
		if self.isSequenceModeling:
			seq_out = self.SequenceModeling2(conv_out, src_mask=None)[0]
		else:
			seq_out = conv_out
		if self.isLSTM:
			seq_out = self.SequenceModeling(conv_out)
		atten_out, text_preds = self.attention(seq_out, text, is_train=is_train,  batch_max_length=8, isVis=isVis, v_value=None)
		return atten_out, text_preds
