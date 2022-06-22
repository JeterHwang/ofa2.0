import random

from elastic_nn.dynamic_layers import DynamicConvLayer, DynamicLinearLayer
from elastic_nn.dynamic_layers import DynamicResNetBottleneckBlock
from utils.layers import IdentityLayer, ResidualBlock
from networks import ResNets
from utils import make_divisible, val2list, MyNetwork

__all__ = ['OFAResNets']


class OFAResNets(ResNets):

	def __init__(self, n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0,
	             depth_list=2, expand_ratio_list=0.25, width_mult_list=1.0, ks_list=3):

		self.depth_list 		= val2list(depth_list)			# e.g. [2, 3, 4]
		self.ks_list 			= val2list(ks_list)				# e.g. [3, 5, 7]
		self.expand_ratio_list 	= val2list(expand_ratio_list)	# e.g. [3, 4, 6]
		self.width_mult_list 	= val2list(width_mult_list)		# e.g. [1.0]
		
		# sort
		self.expand_ratio_list.sort()
		self.ks_list.sort()
		self.depth_list.sort()
		self.width_mult_list.sort()

		# channel(width) of each layer multiplied by width_mult
		input_channel 		= [make_divisible(64 * width_mult, MyNetwork.CHANNEL_DIVISIBLE) for width_mult in self.width_mult_list]
		mid_input_channel 	= [make_divisible(channel // 2, MyNetwork.CHANNEL_DIVISIBLE) for channel in input_channel]

		stage_width_list = ResNets.STAGE_WIDTH_LIST.copy() # [256, 512, 1024, 2048]
		for i, width in enumerate(stage_width_list):
			stage_width_list[i] = [
				make_divisible(width * width_mult, MyNetwork.CHANNEL_DIVISIBLE) for width_mult in self.width_mult_list
			]

		## BASE_DEPTH_LIST = [2, 2, 4, 2]
		n_block_list = [base_depth + max(self.depth_list) for base_depth in ResNets.BASE_DEPTH_LIST]
		
		# stride
		stride_list = [1, 2, 2, 2]

		# build input stem
		input_stem = [
			DynamicConvLayer(
				val2list(3), 
				mid_input_channel, 
				3, 
				stride=2, 
				use_bn=True, 
				act_func='relu'
			),
			ResidualBlock(
				DynamicConvLayer(
					mid_input_channel, 
					mid_input_channel, 
					3, 
					stride=1, 
					use_bn=True, 
					act_func='relu'
				),
				IdentityLayer(
					mid_input_channel, 
					mid_input_channel
				)
			),
			DynamicConvLayer(
				mid_input_channel, 
				input_channel, 
				3, 
				stride=1, 
				use_bn=True, 
				act_func='relu'
			)
		]

		# blocks
		blocks = []
		for d, width, s in zip(n_block_list, stage_width_list, stride_list):
			for i in range(d):
				stride = s if i == 0 else 1 # only the first layer of each block is set to s
				bottleneck_block = DynamicResNetBottleneckBlock(
					input_channel, 
					width, 
					expand_ratio_list	= self.expand_ratio_list,
					kernel_size			= 3, 
					stride				= stride, 
					act_func			= 'relu', 
					downsample_mode		= 'avgpool_conv',
				)
				blocks.append(bottleneck_block)
				input_channel = width
		
		# classifier
		classifier = DynamicLinearLayer(
			input_channel, 
			n_classes, 
			dropout_rate=dropout_rate
		)

		super(OFAResNets, self).__init__(input_stem, blocks, classifier)

		# set bn param
		self.set_bn_param(*bn_param)

		# runtime_depth
		self.input_stem_skipping = 0
		self.runtime_depth = [0] * len(n_block_list) ## ??????

	def set_max_net(self):
		self.set_active_subnet(
			d = max(self.depth_list), 			# e.g. 4
			e = max(self.expand_ratio_list), 	# e.g. 6
			w = len(self.width_mult_list) - 1	# e.g. 0
		)

	def set_active_subnet(self, d=None, e=None, w=None, **kwargs):
		## BASE_DEPTH_LIST = [2, 2, 4, 2]
		depth 			= val2list(d, len(ResNets.BASE_DEPTH_LIST) + 1)
		expand_ratio 	= val2list(e, len(self.blocks))
		width_mult 		= val2list(w, len(ResNets.BASE_DEPTH_LIST) + 2)

		# set expand_ratio of every block
		for block, e in zip(self.blocks, expand_ratio):
			if e is not None:
				block.active_expand_ratio = e


		if width_mult[0] is not None:
			self.input_stem[1].conv.active_out_channel = self.input_stem[0].active_out_channel = \
				self.input_stem[0].out_channel_list[width_mult[0]]
		if width_mult[1] is not None:
			self.input_stem[2].active_out_channel = self.input_stem[2].out_channel_list[width_mult[1]]

		if depth[0] is not None:
			self.input_stem_skipping = (depth[0] != max(self.depth_list))
		for stage_id, (block_idx, d, w) in enumerate(zip(self.grouped_block_index, depth[1:], width_mult[2:])):
			if d is not None:
				self.runtime_depth[stage_id] = max(self.depth_list) - d
			if w is not None:
				for idx in block_idx:
					self.blocks[idx].active_out_channel = self.blocks[idx].out_channel_list[w]

	## Recursively set the avtive subnet
	def sample_active_subnet(self):
		# sample expand ratio
		expand_setting = []
		for block in self.blocks:
			expand_setting.append(random.choice(block.expand_ratio_list))

		# sample depth
		depth_setting = [random.choice([max(self.depth_list), min(self.depth_list)])]
		for stage_id in range(len(ResNets.BASE_DEPTH_LIST)):
			depth_setting.append(random.choice(self.depth_list))

		# sample width_mult
		width_mult_setting = [
			random.choice(list(range(len(self.input_stem[0].out_channel_list)))),
			random.choice(list(range(len(self.input_stem[2].out_channel_list)))),
		]
		for stage_id, block_idx in enumerate(self.grouped_block_index):
			stage_first_block = self.blocks[block_idx[0]]
			width_mult_setting.append(
				random.choice(list(range(len(stage_first_block.out_channel_list))))
			)

		arch_config = {
			'd': depth_setting,
			'e': expand_setting,
			'w': width_mult_setting
		}
		self.set_active_subnet(**arch_config)
		return arch_config

	def get_active_subnet(self, preserve_weight=True):
		input_stem = [self.input_stem[0].get_active_subnet(3, preserve_weight)]
		if self.input_stem_skipping <= 0:
			input_stem.append(ResidualBlock(
				self.input_stem[1].conv.get_active_subnet(self.input_stem[0].active_out_channel, preserve_weight),
				IdentityLayer(self.input_stem[0].active_out_channel, self.input_stem[0].active_out_channel)
			))
		input_stem.append(self.input_stem[2].get_active_subnet(self.input_stem[0].active_out_channel, preserve_weight))
		input_channel = self.input_stem[2].active_out_channel

		blocks = []
		for stage_id, block_idx in enumerate(self.grouped_block_index):
			depth_param = self.runtime_depth[stage_id]
			active_idx = block_idx[:len(block_idx) - depth_param]
			for idx in active_idx:
				blocks.append(self.blocks[idx].get_active_subnet(input_channel, preserve_weight))
				input_channel = self.blocks[idx].active_out_channel
		classifier = self.classifier.get_active_subnet(input_channel, preserve_weight)
		subnet = ResNets(input_stem, blocks, classifier)

		subnet.set_bn_param(**self.get_bn_param())
		return subnet

	def get_active_net_config(self):
		input_stem_config = [self.input_stem[0].get_active_subnet_config(3)]
		if self.input_stem_skipping <= 0:
			input_stem_config.append({
				'name': ResidualBlock.__name__,
				'conv': self.input_stem[1].conv.get_active_subnet_config(self.input_stem[0].active_out_channel),
				'shortcut': IdentityLayer(self.input_stem[0].active_out_channel, self.input_stem[0].active_out_channel),
			})
		input_stem_config.append(self.input_stem[2].get_active_subnet_config(self.input_stem[0].active_out_channel))
		input_channel = self.input_stem[2].active_out_channel

		blocks_config = []
		for stage_id, block_idx in enumerate(self.grouped_block_index):
			depth_param = self.runtime_depth[stage_id]
			active_idx = block_idx[:len(block_idx) - depth_param]
			for idx in active_idx:
				blocks_config.append(self.blocks[idx].get_active_subnet_config(input_channel))
				input_channel = self.blocks[idx].active_out_channel
		classifier_config = self.classifier.get_active_subnet_config(input_channel)
		return {
			'name': ResNets.__name__,
			'bn': self.get_bn_param(),
			'input_stem': input_stem_config,
			'blocks': blocks_config,
			'classifier': classifier_config,
		}
