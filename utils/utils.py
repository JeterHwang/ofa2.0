# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import torch.nn.functional as F
import torch.nn as nn
import torch

from utils.pytorch_utils import get_net_device
from utils.common_tools import DistributedTensor, AverageMeter
from int_quantization.fake_quantize import FakeQuantize, disable_observer
from elastic_nn.dynamic_op import DynamicBatchNorm2d
from elastic_nn.dynamic_intrinsic import DynamicSeparableConvBn2dQuant, DynamicConvBn2dQuant

__all__ = ['set_running_statistics']

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
		if isinstance(m, FakeQuantize) and hasattr(m.activation_post_process, 'min_val'):
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
		if isinstance(m, FakeQuantize) and hasattr(m.activation_post_process, 'min_vals'):
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
			assert isinstance(m, FakeQuantize)
			if hasattr(m.activation_post_process, 'min_val'):
				# print(min_val[name].avg)
				# print(max_val[name].avg)
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