import torch.nn as nn

from utils.layers import ConvLayer, LinearLayer, ResNetBottleneckBlock
from utils.pytorch_modules import make_divisible, MyGlobalAvgPool2d
from utils.my_modules import set_bn_param
__all__ = ['ResNets', 'ResNet50']


class ResNets(nn.Module):
	CHANNEL_DIVISIBLE 	= 8
	BASE_DEPTH_LIST 	= [2, 2, 4, 2]
	STAGE_WIDTH_LIST 	= [256, 512, 1024, 2048]

	def __init__(self, input_stem, blocks, classifier):
		super(ResNets, self).__init__()

		self.input_stem 		= nn.ModuleList(input_stem) ## pass from outside 
		self.max_pooling 		= nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
		self.blocks 			= nn.ModuleList(blocks)
		self.global_avg_pool 	= MyGlobalAvgPool2d(keep_dim=False)
		self.classifier 		= classifier				## pass from outside

	def forward(self, x):
		for layer in self.input_stem:
			x = layer(x)
		x = self.max_pooling(x)
		for block in self.blocks:
			x = block(x)
		x = self.global_avg_pool(x)
		x = self.classifier(x)
		return x

	def set_bn_param(self, momentum, eps, gn_channel_per_group=None):
		set_bn_param(self, momentum, eps, gn_channel_per_group)


class ResNet50(ResNets):

	def __init__(
		self, 
		n_classes		= 1000, 
		width_mult		= 1.0, 
		bn_param		= (0.1, 1e-5), 
		dropout_rate	= 0,
	    expand_ratio	= None, 
		depth_param		= None
	):
		expand_ratio = 0.25 if expand_ratio is None else expand_ratio

		input_channel = make_divisible(64 * width_mult, ResNets.CHANNEL_DIVISIBLE)
		stage_width_list = ResNets.STAGE_WIDTH_LIST.copy()
		for i, width in enumerate(stage_width_list):
			stage_width_list[i] = make_divisible(width * width_mult, ResNets.CHANNEL_DIVISIBLE)

		depth_list = [3, 4, 6, 3]
		if depth_param is not None:
			for i, depth in enumerate(ResNets.BASE_DEPTH_LIST):
				depth_list[i] = depth + depth_param

		stride_list = [1, 2, 2, 2]

		# build input stem
		input_stem = [ConvLayer(3, input_channel, kernel_size=7, stride=2, use_bn=True, act_func='relu', ops_order='weight_bn_act',)]

		# blocks
		blocks = []
		for d, width, s in zip(depth_list, stage_width_list, stride_list):
			for i in range(d):
				stride = s if i == 0 else 1
				bottleneck_block = ResNetBottleneckBlock(
					input_channel, 
					width, 
					kernel_size=3, 
					stride=stride, 
					expand_ratio=expand_ratio,
					act_func='relu', 
					downsample_mode='conv',
				)
				blocks.append(bottleneck_block)
				input_channel = width
		# classifier
		classifier = LinearLayer(input_channel, n_classes, dropout_rate=dropout_rate)
		
		super(ResNet50, self).__init__(input_stem, blocks, classifier)

		# set bn param
		self.set_bn_param(*bn_param)

