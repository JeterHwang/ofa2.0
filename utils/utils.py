# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm
from int_quantization.lsq.lsq import LsqQuan, EMA

from utils.pytorch_utils import get_net_device
from utils.common_tools import DistributedTensor, AverageMeter, DistributedMetric, get_same_padding
from utils.quant import MyMinMaxObserver
from int_quantization.fake_quantize import FakeQuantize, disable_observer
from elastic_nn.dynamic_op import DynamicBatchNorm2d
from elastic_nn.dynamic_intrinsic import DynamicSeparableConvBn2dQuant, DynamicConvBn2dQuant

__all__ = ['set_running_statistics', 'set_activation', 'train_activation']

def set_running_statistics(model, data_loader, distributed=False):
	bn_mean = {}
	bn_var = {}
	min_val = {}
	max_val = {}
	forward_model = copy.deepcopy(model)
	for name, m in forward_model.module.named_modules():
		if isinstance(m, DynamicSeparableConvBn2dQuant) or isinstance(m, DynamicConvBn2dQuant):
			if distributed:
				bn_mean[name] = DistributedTensor(name + "#mean")
				bn_var[name] = DistributedTensor(name + "#var")
			else:
				bn_mean[name] = AverageMeter()
				bn_var[name] = AverageMeter()
			def new_forward(bn, mean_est, var_est):
				def lambda_forward(x):
					batch_mean = (x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
					)  # 1, C, 1, 1
					batch_var = (x - batch_mean) * (x - batch_mean)
					batch_var = (batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True))

					batch_mean = torch.squeeze(batch_mean)
					batch_var = torch.squeeze(batch_var)
					
					mean_est.update(batch_mean.data, x.size(0))
					var_est.update(batch_var.data, x.size(0))

					# bn forward using calculated mean & var
					_feature_dim = batch_mean.size(0)
					return F.batch_norm(
					    x,
					    batch_mean,
					    batch_var,
					    bn.weight[:_feature_dim],
					    bn.bias[:_feature_dim],
					    False,
					    0.0,
					    bn.eps,
					)
				return lambda_forward
			m.bn.forward = new_forward(m.bn, bn_mean[name], bn_var[name])
			m.track_bn_stat = True
		elif isinstance(m, FakeQuantize) and hasattr(m.activation_post_process, 'min_val'):
			if distributed:
				min_val[name] = DistributedTensor(name + "#mean")
				max_val[name] = DistributedTensor(name + "#var")
			else:
				min_val[name] = AverageMeter()
				max_val[name] = AverageMeter()
			def new_forward(quant, min_est, max_est):
				def lambda_forward(x_orig):
					x = x_orig.detach()  # avoid keeping autograd tape
					x = x.to(quant.min_val.dtype)
					new_min = torch.min(x)
					new_max = torch.max(x)
					min_est.update(new_min.data)
					max_est.update(new_max.data)
					quant.min_val.data.copy_(new_min.data)
					quant.max_val.data.copy_(new_max.data)
					return x_orig
				return lambda_forward
			m.activation_post_process.forward = new_forward(m.activation_post_process, min_val[name], max_val[name])
		elif isinstance(m, FakeQuantize) and hasattr(m.activation_post_process, 'min_vals'):
			if distributed:
				min_val[name] = DistributedTensor(name + "#mean")
				max_val[name] = DistributedTensor(name + "#var")
			else:
				min_val[name] = AverageMeter()
				max_val[name] = AverageMeter()
			def new_forward(quant, min_est, max_est):
				def lambda_forward(x_orig):
					x = x_orig.detach()  # avoid keeping autograd tape
					x = x.to(quant.min_vals.dtype)
					x_dim = x.size()

					new_axis_list = list(range(len(x_dim)))
					new_axis_list[quant.ch_axis] = 0
					new_axis_list[0] = quant.ch_axis
					y = x.permute(tuple(new_axis_list))
					y = torch.flatten(y, start_dim=1)
        
					out_channel = y.size(0)
					assert out_channel <= quant.max_channel

					new_min = torch.min(y, 1)[0]
					new_max = torch.max(y, 1)[0]
					min_est.update(new_min.data)
					max_est.update(new_max.data)

					if quant.min_vals.numel() == 0 or quant.max_vals.numel() == 0:
						quant.min_vals.resize_(quant.max_channel)
						quant.max_vals.resize_(quant.max_channel)
						quant.min_vals = torch.zeros_like(quant.min_vals)
						quant.max_vals = torch.zeros_like(quant.max_vals)
					quant.min_vals.data[:out_channel].copy_(new_min.data)
					quant.max_vals.data[:out_channel].copy_(new_max.data)
					return x_orig
				return lambda_forward
			m.activation_post_process.forward = new_forward(m.activation_post_process, min_val[name], max_val[name])
		elif isinstance(m, EMA):
			if distributed:
				min_val[name] = DistributedTensor(name + "#mean")
				max_val[name] = DistributedTensor(name + "#var")
			else:
				min_val[name] = AverageMeter()
				max_val[name] = AverageMeter()
			def new_forward(ema, min_est, max_est):
				def lambda_forward(x):
					xmax = torch.max(x.detach(), dim=1)[0].mean()
					xmin = torch.min(x.detach(), dim=1)[0].mean()
					min_est.update(xmin.data)
					max_est.update(xmax.data)
					return xmax, xmin
				return lambda_forward
			m.forward = new_forward(m, min_val[name], max_val[name])
	
	with torch.no_grad():
		for images, labels in data_loader:
			images = images.to(get_net_device(forward_model))
			forward_model(images)
	
	for name, m in model.module.named_modules():
		if name in bn_mean and bn_mean[name].count > 0:
			assert isinstance(m, DynamicSeparableConvBn2dQuant) or isinstance(m, DynamicConvBn2dQuant)
			feature_dim = bn_mean[name].avg.size(0)
			m.bn.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
			m.bn.running_var.data[:feature_dim].copy_(bn_var[name].avg)
		if name in min_val and min_val[name].count > 0:
			if isinstance(m, FakeQuantize):
				if hasattr(m.activation_post_process, 'min_val'):
					m.activation_post_process.min_val.resize_(min_val[name].avg.shape)
					m.activation_post_process.max_val.resize_(max_val[name].avg.shape)
					m.activation_post_process.min_val.data.copy_(min_val[name].avg)
					m.activation_post_process.max_val.data.copy_(max_val[name].avg)
				else:
					feature_dim = min_val[name].avg.size(0)
					assert feature_dim <= m.max_channel
					m.activation_post_process.min_vals.resize_(m.max_channel)
					m.activation_post_process.max_vals.resize_(m.max_channel)
					m.activation_post_process.min_vals = torch.zeros_like(m.activation_post_process.min_vals)
					m.activation_post_process.max_vals = torch.zeros_like(m.activation_post_process.max_vals)
					m.activation_post_process.min_vals.data[:feature_dim].copy_(min_val[name].avg)
					m.activation_post_process.max_vals.data[:feature_dim].copy_(max_val[name].avg)
			elif isinstance(m, EMA):
				m.min_val.resize_(min_val[name].avg.shape)
				m.max_val.resize_(max_val[name].avg.shape)
				m.min_val.data.copy_(min_val[name].avg)
				m.max_val.data.copy_(max_val[name].avg)

def set_activation(model, data_loader, distributed=False):
	Xmin = {}
	Xmax = {}
	forward_model = copy.deepcopy(model)
	forward_model.module.set_active_subnet(**{
		'iqw': 'fp32',
		'sqw': 'fp32', 
		'iqa': 'fp32', 
		'sqa': 'fp32',
	})
	for name, m in forward_model.module.named_modules():
		if isinstance(m, DynamicSeparableConvBn2dQuant) or isinstance(m, DynamicConvBn2dQuant):
			Xmin[name] = torch.tensor([])
			Xmax[name] = torch.tensor([])
			m.observer.add_module('act_observer', MyMinMaxObserver())
			def new_forward(observer, min_est, max_est):
				def lambda_forward(x_orig):
					x = x_orig.detach()  # avoid keeping autograd tape
					x = x.to(observer.min_val.dtype)
					min_val = min_est
					max_val = max_est
					
					assert torch.isnan(min_val).sum() == 0
					assert torch.isnan(max_val).sum() == 0
					
					batch_mean = x.mean()
					batch_var = ((x - batch_mean) * (x - batch_mean)).mean()
					batch_std = torch.sqrt(batch_var)
					new_min = batch_mean - 3 * batch_std
					new_max = batch_mean + 3 * batch_std
					# new_min = torch.min(x)
					# new_max = torch.max(x)

					if min_val.numel() == 0 or max_val.numel() == 0:
						min_val = new_min
						max_val = new_max
					else:
						min_val = min_val + observer.averaging_constant * (new_min - min_val)
						max_val = max_val + observer.averaging_constant * (new_max - max_val)
					
					min_est.resize_(min_val.shape)
					max_est.resize_(max_val.shape)
					min_est.copy_(min_val)
					max_est.copy_(max_val)
				return lambda_forward
			m.observer.act_observer.forward = new_forward(m.observer.act_observer, Xmin[name], Xmax[name])
			
	with torch.no_grad():
		for images, labels in data_loader:
			images = images.to(get_net_device(forward_model))
			forward_model(images)

	for name, m in model.module.named_modules():
		if name in Xmin and name in Xmax and Xmin[name].numel() != 0 and Xmax[name].numel() != 0:
			assert isinstance(m, DynamicSeparableConvBn2dQuant) or isinstance(m, DynamicConvBn2dQuant)
			for mod in m.act_quant_list:
				if isinstance(mod, LsqQuan):
					mod.init_act(Xmin[name], Xmax[name])

def train_activation(model, data_loader, distributed=False):
	step = {}
	offset = {}
	# loss = DistributedMetric('loss', torch.distributed.get_world_size())
	loss = AverageMeter()
	forward_model = copy.deepcopy(model)
	optimizer = torch.optim.Adam(forward_model.parameters(), 1e-3)

	for name, m in forward_model.named_modules():
		if isinstance(m, DynamicSeparableConvBn2dQuant):
			def new_forward(mod, loss_est):
				def lambda_forward(x):
					x = x.detach()
					for act_quant in mod.act_quant_list:
						if isinstance(act_quant, LsqQuan):
							loss = F.mse_loss(act_quant(x), x)
							loss_est.update(loss.item())
							# loss_est.append(loss)
							loss.backward()
					kernel_size = mod.active_kernel_size
					in_channel = x.size(1)
					filters = mod.get_active_filter(in_channel, kernel_size).contiguous()
					weight_fake_quant = mod.weight_quant_list[mod.weight_quant_mapping[mod.active_weight_quant]] if mod.weight_quant_mapping[mod.active_weight_quant] is not None else None 
					padding = get_same_padding(kernel_size)
        
					assert mod.bn.running_var is not None
					conv = F.conv2d(
						x, 
					    # filters if weight_fake_quant is None else weight_fake_quant(filters), 
					    filters,
						None, # zero_bias, 
					    mod.stride, 
					    padding, 
					    mod.dilation, 
					    in_channel
					)
					output = mod.bn(conv)
					return output
				return lambda_forward
			m.forward = new_forward(m, loss)
		elif isinstance(m, DynamicConvBn2dQuant):
			def new_forward(mod, loss_est):
				def lambda_forward(x):
					x = x.detach()
					for act_quant in mod.act_quant_list:
						if isinstance(act_quant, LsqQuan):
							loss = F.mse_loss(act_quant(x), x)
							# loss_est.append(loss)
							loss_est.update(loss.item())
							loss.backward()
					out_channel = mod.active_out_channel
					in_channel = x.size(1)
					filters = mod.get_active_filter(out_channel, in_channel).contiguous()
					weight_fake_quant = mod.weight_quant_list[mod.weight_quant_mapping[mod.active_weight_quant]] if mod.weight_quant_mapping[mod.active_weight_quant] is not None else None
					padding = get_same_padding(mod.kernel_size)
        
					assert mod.bn.running_var is not None
					conv = F.conv2d(
						x, 
					    # filters if weight_fake_quant is None else weight_fake_quant(filters), 
					    filters,
						None, # zero_bias, 
					    mod.stride, 
					    padding, 
					    mod.dilation, 
					    1
					)
					output = mod.bn(conv)
					return output
				return lambda_forward
			m.forward = new_forward(m, loss)
		
	with tqdm(total=len(data_loader), desc='Training Initial Parameters', disable=torch.distributed.get_rank() != 0) as t:
		forward_model = forward_model.train()
		for images, labels in data_loader:
			images = images.to(get_net_device(forward_model))
			forward_model(images)
			
			# total_loss = 0
			# for L in loss:
			# 	total_loss += L
			# total_loss.backward()
			
			optimizer.step()
			optimizer.zero_grad()
			t.set_postfix({'Loss' : loss.avg})
			t.update(1)
			
	for name, m in forward_model.named_modules():
		if 'act_quant_list' in name and isinstance(m, LsqQuan):
			step[name] = m.s.data
			offset[name] = m.offset.data

	for name, m in model.named_modules():
		if name in step and name in offset:
			assert isinstance(m, LsqQuan)
			# print(m.s.data, m.offset.data)
			# print(step[name], offset[name])
			m.s.data.copy_(step[name])
			m.offset.data.copy_(offset[name])