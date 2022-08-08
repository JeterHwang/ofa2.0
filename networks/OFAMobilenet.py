# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import random

from elastic_nn.dynamic_layers import DynamicMBConvLayerQuant, DynamicMBConvLayer
from utils.layers import ConvLayer, IdentityLayer, LinearLayer, MBConvLayer, ResidualBlock
from networks.MobilenetV3 import MobileNetV3
from utils import make_divisible, val2list, MyNetwork

__all__ = ['OFAMobileNetV3']


class OFAMobileNetV3(MobileNetV3):

	def __init__(
		self, 
		n_classes			= 1000, 
		bn_param			= (0.1, 1e-5), 
		dropout_rate		= 0.1, 		## only used in last classsifier !!
		base_stage_width	= None, 
		width_mult			= 1.0,
	    ks_list				= 3, 
		expand_ratio_list	= 6, 
		depth_list			= 4,
		weight_quant_list   = 'int4_per_channel',
		act_quant_list		= 'int8',
	):

		self.width_mult 		= width_mult ## Width of every stage will be scaled by this factor
		self.ks_list 			= val2list(ks_list, 1)
		self.expand_ratio_list 	= val2list(expand_ratio_list, 1)
		self.depth_list 		= val2list(depth_list, 1)
		self.weight_quant_list  = val2list(weight_quant_list, 1)
		self.act_quant_list		= val2list(act_quant_list, 1)

		self.ks_list.sort()
		self.expand_ratio_list.sort()
		self.depth_list.sort()

		base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]
		final_expand_width = make_divisible(base_stage_width[-2] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)    # 960
		last_channel = make_divisible(base_stage_width[-1] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)			# 1280
		width_list = []
		for base_width in base_stage_width[:-2]:
			width = make_divisible(base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
			width_list.append(width)
		
		stride_stages = [1, 2, 2, 2, 1, 2]
		
		## Mobilenet related parameters
		act_stages = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
		se_stages = [False, False, True, False, True, True]
		n_block_list = [1] + [max(self.depth_list)] * 5

		input_channel, first_block_dim = width_list[0], width_list[1]
		########################## NON DYNAMIC LAYERS ###########################
		# first conv layer														#
		first_conv = ConvLayer(													#
			3, 																	#
			input_channel, 														#
			kernel_size=3, 														#
			stride=2, 															#
			act_func='h_swish'													#
		)																		#
		first_block_conv = MBConvLayer(											#
			in_channels	= input_channel, 	# 16								#
			out_channels= first_block_dim, 	# 16								#
			kernel_size	= 3, 													#
			stride		= stride_stages[0],										#
			expand_ratio= 1, 													#
			act_func	= act_stages[0], 										#
			use_se		= se_stages[0],											#
		)																		#
		first_block = ResidualBlock(											#
			first_block_conv,													#
			IdentityLayer(first_block_dim, first_block_dim) if \
				input_channel == first_block_dim else None,						#			
		)																		#
		#########################################################################

		# inverted residual blocks
		self.block_group_info = []
		blocks 			= [first_block]
		_block_index 	= 1
		feature_dim 	= first_block_dim

		for width, n_block, s, act_func, use_se in zip(width_list[2:], n_block_list[1:],
		                                               stride_stages[1:], act_stages[1:], se_stages[1:]):
			## Store the block index in this stage
			self.block_group_info.append([_block_index + i for i in range(n_block)])
			_block_index += n_block

			output_channel = width
			for i in range(n_block):
				if i == 0:
					stride = s
				else:
					stride = 1
				
				mobile_inverted_conv = DynamicMBConvLayerQuant(
					in_channel_list		= val2list(feature_dim), 
					out_channel_list	= val2list(output_channel),
					kernel_size_list	= ks_list, 
					expand_ratio_list	= expand_ratio_list,
					stride				= stride, 
					act_func			= act_func, 
					use_se				= use_se,
					##### QUANT ######
					inv_weight_quant    = self.weight_quant_list,
        			inv_act_quant       = self.act_quant_list,
        			sep_weight_quant    = self.weight_quant_list,
        			sep_act_quant       = self.act_quant_list,   
				)
				
				if stride == 1 and feature_dim == output_channel:   ## i > 0
					shortcut = IdentityLayer(feature_dim, feature_dim)
				else: 												## i = 0
					shortcut = None
				
				blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
				feature_dim = output_channel
		
		# final expand layer, feature mix layer & classifier
		final_expand_layer = ConvLayer(
			feature_dim, 			## 160
			final_expand_width, 	## 960
			kernel_size=1, 
			act_func='h_swish'
		)
		feature_mix_layer = ConvLayer(
			final_expand_width, 	## 960
			last_channel, 			## 1280
			kernel_size=1, 
			bias=False, 
			use_bn=False, 
			act_func='h_swish',
		)

		classifier = LinearLayer(
			last_channel, 
			n_classes, 
			dropout_rate=dropout_rate
		)

		super(OFAMobileNetV3, self).__init__(
			first_conv, 
			blocks, 
			final_expand_layer, 
			feature_mix_layer, 
			classifier
		)

		# set bn param
		self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

		# runtime_depth (only count for ELASTIC layers)
		self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

	""" MyNetwork required methods """

	@staticmethod
	def name():
		return 'OFAMobileNetV3'

	def forward(self, x):
		# first conv
		x = self.first_conv(x)
		# first block
		x = self.blocks[0](x)
		# blocks
		for stage_id, block_idx in enumerate(self.block_group_info):
			depth = self.runtime_depth[stage_id]
			active_idx = block_idx[:depth]
			for idx in active_idx:
				x = self.blocks[idx](x)
		x = self.final_expand_layer(x)
		x = x.mean(3, keepdim=True).mean(2, keepdim=True)  	# global average pooling
		x = self.feature_mix_layer(x)
		x = x.view(x.size(0), -1) 						 	# flatten
		x = self.classifier(x)
		return x

	@property
	def module_str(self):
		_str = self.first_conv.module_str + '\n'
		_str += self.blocks[0].module_str + '\n'

		for stage_id, block_idx in enumerate(self.block_group_info):
			depth = self.runtime_depth[stage_id]
			active_idx = block_idx[:depth]
			for idx in active_idx:
				_str += self.blocks[idx].module_str + '\n'

		_str += self.final_expand_layer.module_str + '\n'
		_str += self.feature_mix_layer.module_str + '\n'
		_str += self.classifier.module_str + '\n'
		return _str

	@property
	def config(self):
		return {
			'name'				: OFAMobileNetV3.__name__,
			'bn'				: self.get_bn_param(),
			'first_conv'		: self.first_conv.config,
			'blocks'			: [block.config for block in self.blocks],
			'final_expand_layer': self.final_expand_layer.config,
			'feature_mix_layer'	: self.feature_mix_layer.config,
			'classifier'		: self.classifier.config,
		}

	@property
	def grouped_block_index(self):
		return self.block_group_info

	def load_state_dict(self, state_dict, **kwargs):
		model_dict = self.state_dict()
		for key in state_dict:
			if '.mobile_inverted_conv.' in key:
				new_key = key.replace('.mobile_inverted_conv.', '.conv.')
			elif 'module.' in key:
				new_key = key.replace('module.', '')
			else:
				new_key = key
			if 'min_val' in new_key or 'max_val' in new_key:
				continue	
			
			if new_key in model_dict:
				pass
			elif '.bn.bn.' in new_key:
				new_key = new_key.replace('.bn.bn.', '.bn.')
			elif '.conv.conv.weight' in new_key:
				new_key = new_key.replace('.conv.conv.weight', '.conv.weight')
			elif '.linear.linear.' in new_key:
				new_key = new_key.replace('.linear.linear.', '.linear.')
			##############################################################################
			elif '.linear.' in new_key:
				new_key = new_key.replace('.linear.', '.linear.linear.')
			elif 'bn.' in new_key:
				new_key = new_key.replace('bn.', 'bn.bn.')
			elif 'conv.weight' in new_key:
				new_key = new_key.replace('conv.weight', 'conv.conv.weight')
			else:
				raise ValueError(new_key)
			assert new_key in model_dict, '%s' % new_key
			model_dict[new_key] = state_dict[key]
		super(OFAMobileNetV3, self).load_state_dict(model_dict)

	""" set, sample and get active sub-networks """

	@property
	def max_subnet(self):
		return {
			'ks' : max(self.ks_list), 
			'e'	: max(self.expand_ratio_list), 
			'd'	: max(self.depth_list),
			'iqw' : self.weight_quant_list[0],
			'sqw' : self.weight_quant_list[0],
			'iqa' : self.act_quant_list[0],
			'sqa' : self.act_quant_list[0],
		}
	@property
	def min_subnet(self):
		return {
			'ks' : min(self.ks_list), 
			'e'	: min(self.expand_ratio_list), 
			'd'	: min(self.depth_list),
			'iqw' : self.weight_quant_list[-1],
			'sqw' : self.weight_quant_list[-1],
			'iqa' : self.act_quant_list[-1],
			'sqa' : self.act_quant_list[-1],
		}

	def set_constraint(self, include_list, constraint_type='depth'):
		if constraint_type == 'depth':
			self.__dict__['_depth_include_list'] = include_list.copy()
		elif constraint_type == 'expand_ratio':
			self.__dict__['_expand_include_list'] = include_list.copy()
		elif constraint_type == 'kernel_size':
			self.__dict__['_ks_include_list'] = include_list.copy()
		elif constraint_type == 'weight_quant':
			self.__dict__['_weightQ_include_list'] = include_list.copy()
		elif constraint_type == 'act_quant':
			self.__dict__['_actQ_include_list'] = include_list.copy()
		else:
			raise NotImplementedError

	def clear_constraint(self):
		self.__dict__['_depth_include_list'] = None
		self.__dict__['_expand_include_list'] = None
		self.__dict__['_ks_include_list'] = None
		self.__dict__['_weightQ_include_list'] = None
		self.__dict__['_actQ_include_list'] = None

	def set_active_subnet(self, ks=None, e=None, d=None, iqw=None, sqw=None, iqa=None, sqa=None, **kwargs):
		ks 				= val2list(ks, len(self.blocks) - 1)
		expand_ratio 	= val2list(e, len(self.blocks) - 1)
		depth 			= val2list(d, len(self.block_group_info))
		inv_qw		    = val2list(iqw, len(self.blocks) - 1) 
		sep_qw			= val2list(sqw, len(self.blocks) - 1)
		inv_qa			= val2list(iqa, len(self.blocks) - 1)
		sep_qa			= val2list(sqa, len(self.blocks) - 1)

		for block, k, e, iqw, iqa, sqw, sqa in zip(self.blocks[1:], ks, expand_ratio, inv_qw, inv_qa, sep_qw, sep_qa):
			if k is not None:
				block.conv.active_kernel_size = k
			if e is not None:
				block.conv.active_expand_ratio = e
			if iqw is not None:
				block.conv.active_inv_wquant = iqw
			if iqa is not None:
				block.conv.active_inv_aquant = iqa
			if sqw is not None:
				block.conv.active_sep_wquant = sqw
			if sqa is not None:
				block.conv.active_sep_aquant = sqa

		for i, d in enumerate(depth):
			if d is not None:
				self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

	def sample_active_subnet(self):
		# If constraint is set, use constraint
		# Otherwirse, use current configurations
		ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
			else self.__dict__['_ks_include_list']
		expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
			else self.__dict__['_expand_include_list']
		depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None \
			else self.__dict__['_depth_include_list']
		weight_quant_candidates = self.weight_quant_list if self.__dict__.get('_weightQ_include_list', None) is None \
			else self.__dict__['_weightQ_include_list']
		act_quant_candidates = self.act_quant_list if self.__dict__.get('_actQ_include_list', None) is None \
			else self.__dict__['_actQ_include_list']

		# sample kernel size
		ks_setting = []
		if not isinstance(ks_candidates[0], list):
			ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
		for k_set in ks_candidates:
			k = random.choice(k_set)
			ks_setting.append(k)

		# sample expand ratio
		expand_setting = []
		if not isinstance(expand_candidates[0], list):
			expand_candidates = [expand_candidates for _ in range(len(self.blocks) - 1)]
		for e_set in expand_candidates:
			e = random.choice(e_set)
			expand_setting.append(e)

		# sample depth
		depth_setting = []
		if not isinstance(depth_candidates[0], list):
			depth_candidates = [depth_candidates for _ in range(len(self.block_group_info))]
		for d_set in depth_candidates:
			d = random.choice(d_set)
			depth_setting.append(d)

		# sample quant type
		if not isinstance(weight_quant_candidates[0], list):
			weight_quant_candidates = [weight_quant_candidates for _ in range(len(self.blocks) - 1)]
		if not isinstance(act_quant_candidates[0], list):
			act_quant_candidates = [act_quant_candidates for _ in range(len(self.blocks) - 1)]
		inv_wq_setting, inv_aq_setting, sep_wq_setting, sep_aq_setting = [], [], [], []
		for wq_set, aq_set in zip(weight_quant_candidates, act_quant_candidates):
			wq1, wq2 = random.choice(wq_set), random.choice(wq_set)
			aq1, aq2 = random.choice(aq_set), random.choice(aq_set)
			inv_wq_setting.append(wq1)
			inv_aq_setting.append(aq1)
			sep_wq_setting.append(wq2)
			sep_aq_setting.append(aq2)	
		
		self.set_active_subnet(
			ks_setting, 
			expand_setting, 
			depth_setting, 
			inv_wq_setting, 
			sep_wq_setting, 
			inv_aq_setting, 
			sep_aq_setting
		)
		
		return {
			'ks': ks_setting,
			'e': expand_setting,
			'd': depth_setting,
			'iqw': inv_wq_setting,
			'sqw': sep_wq_setting,
			'iqa': inv_aq_setting,
			'sqa': sep_aq_setting,
		}

	def get_active_subnet(self, preserve_weight=True):
		first_conv = copy.deepcopy(self.first_conv)
		blocks = [copy.deepcopy(self.blocks[0])]

		final_expand_layer = copy.deepcopy(self.final_expand_layer)
		feature_mix_layer = copy.deepcopy(self.feature_mix_layer)
		classifier = copy.deepcopy(self.classifier)

		input_channel = blocks[0].conv.out_channels
		# blocks
		for stage_id, block_idx in enumerate(self.block_group_info):
			depth = self.runtime_depth[stage_id]
			active_idx = block_idx[:depth]
			stage_blocks = []
			for idx in active_idx:
				stage_blocks.append(ResidualBlock(
					self.blocks[idx].conv.get_active_subnet(input_channel, preserve_weight),
					copy.deepcopy(self.blocks[idx].shortcut)
				))
				input_channel = stage_blocks[-1].conv.out_channels
			blocks += stage_blocks

		_subnet = MobileNetV3(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
		_subnet.set_bn_param(**self.get_bn_param())
		return _subnet

	def get_active_net_config(self):
		# first conv
		first_conv_config 			= self.first_conv.config
		first_block_config 			= self.blocks[0].config
		final_expand_config 		= self.final_expand_layer.config
		feature_mix_layer_config 	= self.feature_mix_layer.config
		classifier_config 			= self.classifier.config

		block_config_list = [first_block_config]
		input_channel = first_block_config['conv']['out_channels']
		for stage_id, block_idx in enumerate(self.block_group_info):
			depth = self.runtime_depth[stage_id]
			active_idx = block_idx[:depth]
			stage_blocks = []
			for idx in active_idx:
				stage_blocks.append({
					'name': ResidualBlock.__name__,
					'conv': self.blocks[idx].conv.get_active_subnet_config(input_channel),
					'shortcut': self.blocks[idx].shortcut.config if self.blocks[idx].shortcut is not None else None,
				})
				input_channel = self.blocks[idx].conv.active_out_channel
			block_config_list += stage_blocks

		return {
			'name': MobileNetV3.__name__,
			'bn': self.get_bn_param(),
			'first_conv': first_conv_config,
			'blocks': block_config_list,
			'final_expand_layer': final_expand_config,
			'feature_mix_layer': feature_mix_layer_config,
			'classifier': classifier_config,
		}

	""" Width Related Methods """

	def re_organize_middle_weights(self, expand_ratio_stage=0):
		for block in self.blocks[1:]:
			block.conv.re_organize_middle_weights(expand_ratio_stage)
