import torch.nn as nn
import torch
import torch.nn.functional as F

from PVAM.encoder import *
from PVAM.decoder import *
from PVAM.Trans_atten import *


class LPR_model_trans_baseline(nn.Module):

	def __init__(self, nc, nclass, imgW=96, imgH=32, K=8, n_position=192, head=2, inner=512, model_dim = 128, isSeqModel=True):
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
