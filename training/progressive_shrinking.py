# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
import copy
import os
from typing import Dict, List
from unittest import result
import numpy as np
import torch.nn as nn
import random
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import AverageMeter, DistributedMetric, cross_entropy_loss_with_soft_target
from utils import list_mean, subset_mean, val2list, accuracy, MyRandomResizedCrop
from utils import set_running_statistics, profile_quant, init_lsq, set_activation, train_activation
from utils import disable_bn_stats, enable_bn_stats, sync_bn, sync_fake_quant
from int_quantization.fake_quantize import disable_observer, enable_observer


__all__ = [
	'validate', 
	'validate_all_settings',
    'train_one_epoch', 
    'train', 
    'load_pretrained_model',
	'load_model',
	'train_elastic_depth', 
    'train_elastic_expand', 
    'train_elastic_width_mult',
	'train_elastic_bit',
]
## CALLED BY VALIDATE_ALL_SETTINGS
def validate(
	dynamic_net,
	test_criterion,
	run_config,
	args,
	epoch		= 0, 	## Only used for printing
	is_test		= False, 
	run_str		= '', 
	data_loader	= None, 
	no_logs		= False, 
	train_mode	= False
):
    # if not isinstance(dynamic_net, nn.DataParallel):
    #     dynamic_net = nn.DataParallel(dynamic_net)

	if data_loader is None:
		data_loader = run_config.test_loader if is_test else run_config.valid_loader

	dynamic_net.eval()

	losses = DistributedMetric('val_loss', args.world_size)			## Loss
	metric_dict = get_metric_dict(args.world_size) ## Accuracy

	distributed = isinstance(dynamic_net, nn.parallel.DistributedDataParallel)
	if distributed:
		rank = torch.distributed.get_rank()
    
	with torch.no_grad():
		with tqdm(total=len(data_loader), desc='Validate Epoch #{} {}'.format(epoch + 1, run_str), disable=(rank != 0)) as t:
			for i, (images, labels) in enumerate(data_loader):
				images, labels = images.cuda(), labels.cuda()
				# compute output
				output  = dynamic_net(images)
				loss    = test_criterion(output, labels)
				# measure accuracy and record loss
				update_metric(metric_dict, output, labels)

				losses.update(loss, torch.tensor(images.size(0), device=loss.device))
				t.set_postfix({
					'loss': losses.avg,
					**get_metric_vals(metric_dict, return_dict=True),
					'img_size': images.size(2),
				})
				t.update(1)
	return losses.avg, get_metric_vals(metric_dict)

def validate_all_settings(
	dynamic_net, 
	device,
	test_criterion,
	run_config,
	args,
	epoch				= 0, 	## Only used for printing
	is_test				= False, 
	image_size_list		= None,
    ks_list				= None, 
	expand_ratio_list	= None, 
	depth_list			= None, 
	width_mult_list		= None, 
	weight_quant_list   = None,
	act_quant_list		= None,
	additional_setting	= None
):
	dynamic_net.eval()

	## Image Resolution
	if image_size_list is None:
		image_size_list = val2list(run_config.data_provider.image_size, 1)
	## Kernel size, expand ratio, depth, width
	if ks_list is None:
		ks_list = dynamic_net.module.ks_list
	if expand_ratio_list is None:
		expand_ratio_list = dynamic_net.module.expand_ratio_list
	if depth_list is None:
		depth_list = dynamic_net.module.depth_list
	if weight_quant_list is None:
		weight_quant_list = dynamic_net.module.weight_quant_list 
	if act_quant_list is None:
		act_quant_list = dynamic_net.module.act_quant_list
	if width_mult_list is None: ### NOT USED !!
		if 'width_mult_list' in dynamic_net.module.__dict__:
			width_mult_list = list(range(len(dynamic_net.module.width_mult_list)))
		else:
			width_mult_list = [0]

	subnet_settings = []
	for qw in weight_quant_list:
		for qa in act_quant_list:
			for d in [max(depth_list)]:
				for e in [max(expand_ratio_list)]:
					for k in [max(ks_list)]:
						for w in [max(width_mult_list)]:
							for img_size in [max(image_size_list)]:
								subnet_settings.append([{
									'image_size': img_size,
									'd': d,
									'e': e,
									'ks': k,
									'w': w,
									'iqw': qw, 
									'sqw': qw, 
									'iqa': qa, 
									'sqa': qa
								}, 'R%s-D%s-E%s-K%s-W%s-QW-%s-QA-%s' % (img_size, d, e, k, w, qw, qa)])
	if additional_setting is not None:
		subnet_settings += additional_setting

	losses_of_subnets, top1_of_subnets, top5_of_subnets = [], [], []

	valid_log = ''
	for setting, name in subnet_settings:
		# Image resolution
		run_config.data_provider.assign_active_img_size(setting.pop('image_size'))
		# subnet settings
		dynamic_net.module.set_active_subnet(**setting)
		# Clear previous results
		reset_running_statistics(
			net=dynamic_net, 
			run_config=run_config,  
			distributed=False
		)
		# for name, m in dynamic_net.named_modules():
		# 	if name == 'module.blocks.1.conv.point_linear.conv.act_quant_list.0':
		# 		print(m.activation_post_process.min_val, m.activation_post_process.max_val, m.scale, m.zero_point)
		dynamic_net.apply(disable_observer)
		# dynamic_net.apply(disable_bn_stats)
		loss, (top1, top5) = validate(
			dynamic_net,
			test_criterion,
			run_config,
			args,
			epoch	= epoch, 
			is_test	= is_test, 
			run_str	= name, 
		)
		# if torch.distributed.get_rank() == 0:
		# 	bitOps, params = get_cost(dynamic_net, run_config.data_provider.active_img_size)
		# 	print(bitOps, params)
		# torch.distributed.barrier()
		dynamic_net.apply(enable_observer)
		
		# dynamic_net.apply(enable_bn_stats)
		losses_of_subnets.append(loss)
		top1_of_subnets.append(top1)
		top5_of_subnets.append(top5)
		valid_log += '%s (%.3f), ' % (name, top1)

	return list_mean(losses_of_subnets), list_mean(top1_of_subnets), list_mean(top5_of_subnets), valid_log

def validate_one_setting(
	dynamic_net, 
	test_criterion,
	run_config,
	args,
	subnet_setting,
	subnet_str = "",
	image_size = 224,
	epoch = 0,
	is_test = False, 
):
	dynamic_net.eval()
	# Image resolution
	run_config.data_provider.assign_active_img_size(image_size)
	dynamic_net.module.set_active_subnet(**subnet_setting)
	reset_running_statistics(
		net=dynamic_net, 
		run_config=run_config,  
		distributed=False
	)
	dynamic_net.apply(disable_observer)
	# dynamic_net.apply(disable_bn_stats)
	loss, (top1, top5) = validate(
		dynamic_net,
		test_criterion,
		run_config,
		args,
		epoch	= epoch, 
		is_test	= is_test, 
		run_str	= "Validate No.1 Subnet", 
	)
	dynamic_net.apply(enable_observer)
	if torch.distributed.get_rank() == 0:
		print(f"===== No.1 validation accuracy({top1}) =====")

def train_one_epoch(
	dynamic_net, 
	device, 
	criterion, 
	optimizer, 
	run_config, 
	args, 
	epoch, 
	warmup_epochs=0, 
	warmup_lr=0,
	subnet_pool : List=None,
):
	distributed = isinstance(dynamic_net, nn.parallel.DistributedDataParallel)
	if distributed:
		run_config.train_loader.sampler.set_epoch(epoch)
	MyRandomResizedCrop.EPOCH = epoch

	dynamic_net.train()

	# batch size 
	nBatch 		    = len(run_config.train_loader)
	subnet_seed     = 0
	first           = 0
	last            = 100
	data_time 	    = AverageMeter()
	losses 		    = DistributedMetric("train_loss", args.world_size) if distributed else AverageMeter()
	subnet_settings = [dynamic_net.module.sample_active_subnet() for _ in range(args.dynamic_batch_size)]
	subnet_strs     = [get_module_str(dynamic_net, setting) for setting in subnet_settings]
	acc1_of_subnets = [AverageMeter() if not distributed else DistributedMetric("Subnet ACC", args.world_size) for _ in range(len(subnet_settings))]
	result          = {}		 
	# Top1, Top5
	metric_dict     = get_metric_dict(args.world_size)

	with tqdm(total=nBatch, desc='Train Epoch #{}'.format(epoch + 1), disable=distributed and (device != 0)) as t:
		end = time.time()
		for i, (images, labels) in enumerate(run_config.train_loader):
			MyRandomResizedCrop.BATCH = i
			data_time.update(time.time() - end)
			
			images, labels = images.cuda(), labels.cuda()
			target = labels
			# soft target
			if args.kd_ratio > 0:
				args.teacher_model.train()
				with torch.no_grad():
					soft_logits = args.teacher_model(images).detach()
					soft_label = F.softmax(soft_logits, dim=1)
			# clean gradients
			# dynamic_net.zero_grad()
			loss_type = '%.1fkd-%s & ce' % (args.kd_ratio, args.kd_type)
			loss_of_subnets = []
			
			for setting_id, setting in enumerate(subnet_settings):
				dynamic_net.module.set_active_subnet(**setting)
				if (i + 1) % args.gradient_accumulation_steps != 0:
					with dynamic_net.no_sync():
						output = dynamic_net(images)
						if args.kd_ratio == 0:
							loss = criterion(output, labels)
							loss_type = 'ce'
						else:
							if args.kd_type == 'ce':
								kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
							else:
								kd_loss = F.mse_loss(output, soft_logits)
							loss = args.kd_ratio * kd_loss + criterion(output, labels)
						loss_of_subnets.append(loss.item())
						loss = loss / args.gradient_accumulation_steps
						loss.backward()
				else:
					output = dynamic_net(images)
					if args.kd_ratio == 0:
						loss = criterion(output, labels)
						loss_type = 'ce'
					else:
						if args.kd_type == 'ce':
							kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
						else:
							kd_loss = F.mse_loss(output, soft_logits)
						loss = args.kd_ratio * kd_loss + criterion(output, labels)
					loss_of_subnets.append(loss.item())
					loss = loss / args.gradient_accumulation_steps
					loss.backward()

				acc1, acc5 = update_metric(metric_dict, output, target)
				acc1_of_subnets[setting_id].update(acc1)
			
			losses.update(list_mean(loss_of_subnets), torch.tensor(images.size(0), device=images.device))
			
			if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1 == len(run_config.train_loader)):
				nn.utils.clip_grad_norm_(dynamic_net.parameters(), max_norm=10.0, norm_type=2)
				optimizer.step()
				optimizer.zero_grad()
				# Update subnet result
				for setting, module_str, top_1_metric in zip(subnet_settings, subnet_strs, acc1_of_subnets):
					# dynamic_net.module.set_active_subnet(**setting)
					top_1 = top_1_metric.avg.item()
					top_1_metric.reset()
					result.update({module_str : (top_1, setting)})	
					if top_1 > first:
						first = top_1
					if top_1 < last:
						last = top_1
				# Sample subnets
				subnet_seed = int('%d%.3d%.3d' % (epoch * nBatch + i, 1, 0))
				random.seed(subnet_seed)
				subnet_settings, subnet_strs = [], []
				for _ in range(args.dynamic_batch_size):
					subnet_settings.append(dynamic_net.module.sample_active_subnet())
					subnet_strs.append(get_module_str(dynamic_net))
				# for _ in range(args.dynamic_batch_size // 2, args.dynamic_batch_size):
				# 	selected_subnet_setting = random.choice(subnet_pool)
				# 	subnet_settings.append(selected_subnet_setting)
				# 	subnet_strs.append(get_module_str(dynamic_net, selected_subnet_setting))
				# Adjust 
				if epoch < warmup_epochs:
					new_lr = run_config.warmup_adjust_learning_rate(
						optimizer, 
						warmup_epochs * nBatch, 
						nBatch, 
						epoch, 
						i, 
						warmup_lr,
					)
				else:
					new_lr = run_config.adjust_learning_rate(
						optimizer, 
						epoch - warmup_epochs, 
						i, 
						nBatch
					)
			
			t.set_postfix({
				'loss': losses.avg.item(),
				**get_metric_vals(metric_dict, return_dict=True),
				'R': images.size(2),
				'lr': optimizer.param_groups[0]['lr'],
				'loss_type': loss_type,
				'seed': str(subnet_seed),
				'first': first,
				'last' : last,
				'data_time': data_time.avg,
			})
			t.update(1)
			end = time.time()
	
	return losses.avg.item(), get_metric_vals(metric_dict), sorted(list(result.values()), key=lambda x : x[0])

def train(
	dynamic_net, 
	device, 
	train_criterion, 
	test_criterion, 
	optimizer, 
	args, 
	run_config, 
	validate_func=None
):
	distributed = isinstance(dynamic_net, nn.parallel.DistributedDataParallel)

	if validate_func is None:
		validate_func = validate_all_settings

	best_acc = 0
	subnet_pool = [dynamic_net.module.min_subnet, dynamic_net.module.min_subnet]

	for epoch in range(args.start_epoch, args.n_epochs + args.warmup_epochs):
		# if (epoch + 1) > args.num_batch_norm_update_epochs:
		# 	dynamic_net.apply(disable_bn_stats)
		train_loss, (train_top1, train_top5), acc_result = train_one_epoch(
			dynamic_net, 
			device,
			train_criterion,
			optimizer,
			run_config,
			args, 
			epoch, 
			args.warmup_epochs, 
			args.warmup_lr,
			subnet_pool
		)
		if torch.distributed.get_rank() == 0:
			print(f"Total {len(acc_result)} of subnets !!")
		subnet_pool = [res[1] for res in acc_result[:500]]
		dynamic_net.apply(sync_bn)
		dynamic_net.apply(sync_fake_quant)
		# cur_min = 0
		# for _ in range(1):
		# 	min_acc, min_setting = 100, {}
		# 	for res in acc_result:
		# 		if min_acc > res[0] and res[0] >= cur_min:
		# 			min_acc = res[0]
		# 			min_setting = res[1]
		# 	subnet_pool.append(min_setting)
		# 	cur_min = min_acc
		# subnet_pool = [acc_result[res[0]][1] for res in result_sorted[:50]]
		
		# if distributed and torch.distributed.get_rank() == 0:
		# 	bitOps_w, params_w = get_cost(dynamic_net, 224, acc_result[0][1])
		# 	print(f"===== Worst training accuracy({acc_result[0][0]}) =====")
		# 	print(f"Bitops = {bitOps_w} (G)")
		# 	print(f"Param size = {params_w} (MB)")
		# torch.distributed.barrier()
		bitOps_w, params_w = get_cost(dynamic_net, 224, acc_result[-1][1])
		if distributed and torch.distributed.get_rank() == 0:
			print(f"===== Best training accuracy({acc_result[-1][0]}) =====")
			print(f"Bitops = {bitOps_w} (G)")
			print(f"Param size = {params_w} (MB)")
		torch.distributed.barrier()
		validate_one_setting(
			dynamic_net, 
			test_criterion,
			run_config,
			args,
			acc_result[-1][1],
			subnet_str = "",
			image_size = 224,
			epoch = epoch,
			is_test = True
		)
		# for name, mod in dynamic_net.named_parameters():
		# 	if 'coefficient' in name and torch.distributed.get_rank() == 0:
		# 		print(name, mod.data.squeeze())
		if (epoch + 1) % args.validation_frequency == 0:
			val_loss, val_acc, val_acc5, _val_log = validate_func(
				dynamic_net,
				device,
				test_criterion,
				run_config,
				args,
				epoch=epoch, 
				is_test=True, ## changed !!
			)
			# best_acc
			is_best = val_acc > best_acc
			best_acc = max(best_acc, val_acc)
			
			if not distributed or torch.distributed.get_rank() == 0:
				save_model(
					args.save_path,
					dynamic_net,
					optimizer,
					run_config,
					best_acc,
					is_best=is_best
				)
			torch.distributed.barrier()

## LOAD MODEL AND PRINT THE MESSAGE
def load_pretrained_model(model, model_path=None):
	# specify init path
	# print('----- Start Loading Pretrained Model -----')
	init = torch.load(model_path, map_location='cpu')['state_dict']
	model.load_state_dict(init)
	# print('----- Finish Load Pretrained Model -----')

def load_pretrained_layer(model, model_path=None):
	pretrained_dict = torch.load(model_path, map_location='cpu')['state_dict']
	model_dict = model.state_dict()
	new_dict = {}
	with torch.no_grad():
		for key, value in pretrained_dict.items():
			if 'mobile_inverted_conv' in key:
				key = key.replace('mobile_inverted_conv', 'conv')
			if key in model_dict:
				new_dict[key] = value
			elif 'inverted_bottleneck.bn' in key:
				new_dict[key.replace('inverted_bottleneck.bn', 'inverted_bottleneck.conv')] = value
			elif 'depth_conv.bn' in key:
				new_dict[key.replace('depth_conv.bn', 'depth_conv.conv')] = value
			elif 'point_linear.bn' in key:
				new_dict[key.replace('point_linear.bn', 'point_linear.conv')] = value
	model_dict.update(new_dict)
	model.load_state_dict(model_dict)
	model.apply(init_lsq)

def train_elastic_depth(
	dynamic_net,
	device,
	train_criterion,
	test_criterion,
	optimizer,
	args, 
	run_config,
	validate_func_dict
):

	depth_stage_list = dynamic_net.module.depth_list.copy()
	depth_stage_list.sort(reverse=True)
	n_stages 		= len(depth_stage_list) - 1
	current_stage 	= n_stages - 1

	# load pretrained models
	if not args.resume:
		validate_func_dict['depth_list'] = sorted(dynamic_net.module.depth_list)

		load_pretrained_model(dynamic_net.module, model_path=args.ofa_checkpoint_path)
		# validate after loading weights
		print('%.3f\t%.3f\t%.3f\t%s' %validate_all_settings(
			dynamic_net,
			device,
			test_criterion,
			run_config,
			args,
			is_test=True, 
			**validate_func_dict
		), 'valid')
	else:
		load_model(
			args.save_path,
			dynamic_net,
			optimizer,
			args,
			args.model_fname
		)

	# add depth list constraints
	if len(set(dynamic_net.module.ks_list)) == 1 and len(set(dynamic_net.module.expand_ratio_list)) == 1:
		validate_func_dict['depth_list'] = depth_stage_list
	else:
		validate_func_dict['depth_list'] = sorted({min(depth_stage_list), max(depth_stage_list)})

	###################################### Train ######################################
	train(																		  #
		dynamic_net,
		device,
		train_criterion,
		test_criterion,
		optimizer, 
		args,	
		run_config,														  			  #
		lambda net, device, test_criterion, run_config, args, epoch, is_test: validate_all_settings(
			net, 
			device,
			test_criterion,
			run_config,
			args,
			epoch, 
			is_test, 
			**validate_func_dict
		)
	)																				  #
	###################################################################################

def train_elastic_expand(
	dynamic_net,
	device,
	train_criterion,
	test_criterion,
	optimizer, 
	args, 
	run_config,
	validate_func_dict
):

	expand_stage_list = dynamic_net.module.expand_ratio_list.copy()
	expand_stage_list.sort(reverse=True)
	n_stages = len(expand_stage_list) - 1
	current_stage = n_stages - 1

	# load pretrained models
	if not args.resume:
		validate_func_dict['expand_ratio_list'] = sorted(dynamic_net.module.expand_ratio_list)

		load_pretrained_model(dynamic_net.module, model_path=args.ofa_checkpoint_path)
		dynamic_net.module.re_organize_middle_weights(expand_ratio_stage=current_stage)
		print('%.3f\t%.3f\t%.3f\t%s' %validate_all_settings(
			dynamic_net,
			device,
			test_criterion,
			run_config,
			args,
			is_test=True, 
			**validate_func_dict), 
		'valid')
	else:
		load_model(
			args.save_path,
			dynamic_net,
			optimizer,
			args,
			args.model_fname
		)

	if len(set(dynamic_net.module.ks_list)) == 1 and len(set(dynamic_net.module.depth_list)) == 1:
		validate_func_dict['expand_ratio_list'] = expand_stage_list
	else:
		validate_func_dict['expand_ratio_list'] = sorted({min(expand_stage_list), max(expand_stage_list)})

	# train
	train(
		dynamic_net,
		device,
		train_criterion,
		test_criterion,
		optimizer, 
		args,	
		run_config,	
		lambda net, device, test_criterion, run_config, args, epoch, is_test: validate_all_settings(
			net, 
			device,
			test_criterion,
			run_config,
			args,
			epoch, 
			is_test, 
			**validate_func_dict
		)
	)

def train_elastic_bit(
	dynamic_net,
	device,
	train_criterion,
	test_criterion,
	optimizer, 
	args, 
	run_config,
	validate_func_dict
):
	
	validate_func_dict['weight_quant_list'] = ['lsq3_per_channel', 'lsq4_per_channel']
	validate_func_dict['act_quant_list'] = ['lsq4_per_tensor', 'lsq5_per_tensor']
	
	if not args.resume:
		load_pretrained_layer(dynamic_net.module, model_path=args.ofa_checkpoint_path)
		# observe_activation(dynamic_net, run_config=run_config, args=args)
		# train_initialize(dynamic_net.module, run_config=run_config, args=args)
		torch.distributed.barrier()
		print('%.3f\t%.3f\t%.3f\t%s' %validate_all_settings(
			dynamic_net,
			device,
			test_criterion,
			run_config,
			args,
			is_test=True, 
			**validate_func_dict), 
		'valid')
	else:
		load_model(
			args.save_path,
			dynamic_net,
			optimizer,
			args,
			args.model_fname
		)
	train(
		dynamic_net,
		device,
		train_criterion,
		test_criterion,
		optimizer, 
		args,	
		run_config,	
		lambda net, device, test_criterion, run_config, args, epoch, is_test: validate_all_settings(
			net, 
			device,
			test_criterion,
			run_config,
			args,
			epoch, 
			is_test, 
			**validate_func_dict
		)
	)
############### NOT USED ################# 
def train_elastic_width_mult(
	dynamic_net, 
	device,
	train_criterion,
	test_criterion,
	optimizer, 
	args, 
	run_config, 
	validate_func_dict
):

	width_stage_list = dynamic_net.module.width_mult_list.copy()
	width_stage_list.sort(reverse=True)
	n_stages = len(width_stage_list) - 1
	current_stage = n_stages - 1

	if not args.resume:
		load_pretrained_model(dynamic_net.module, model_path=args.ofa_checkpoint_path)
		if current_stage == 0:
			dynamic_net.module.re_organize_middle_weights(expand_ratio_stage=len(dynamic_net.module.expand_ratio_list) - 1)
			try:
				dynamic_net.module.re_organize_outer_weights()
			except Exception:
				pass
	else:
		load_model(
			args.save_path,
			dynamic_net,
			optimizer,
			args,
			args.model_fname
		)

	validate_func_dict['width_mult_list'] = sorted({0, len(width_stage_list) - 1})

	# train
	train(
		dynamic_net,
		device,
		train_criterion,
		test_criterion,
		optimizer, 
		args,	
		run_config,	
		lambda net, device, test_criterion, run_config, args, epoch, is_test: validate_all_settings(
			net, 
			device,
			test_criterion,
			run_config,
			args,
			epoch, 
			is_test, 
			**validate_func_dict
		)
	)


""" metric related """

def get_metric_dict(world_size):
    return {
        'top1': DistributedMetric('top1', world_size),
        'top5': DistributedMetric('top5', world_size),
    }

def update_metric(metric_dict, output, labels):
	acc1, acc5 = accuracy(output, labels, topk=(1, 5))
	metric_dict['top1'].update(acc1[0], torch.tensor(output.size(0), device=acc1[0].device))
	metric_dict['top5'].update(acc5[0], torch.tensor(output.size(0), device=acc5[0].device))
	return acc1.item(), acc5.item()

def get_metric_vals(metric_dict, return_dict=False):
    if return_dict:
        return { key: metric_dict[key].avg for key in metric_dict }
    else:
        return [metric_dict[key].avg for key in metric_dict]

def get_metric_names():
    return 'top1', 'top5'

def get_cost(dynamic_net, image_size=224, setting=None):
	forward_model = copy.deepcopy(dynamic_net)
	forward_model.eval()
	if setting is not None:
		forward_model.module.set_active_subnet(**setting)
	bitOps, params = profile_quant(forward_model, (1, 3, image_size, image_size))
	# del forward_model
	return bitOps, params

def get_module_str(dynamic_net, setting=None):
	forward_model = copy.deepcopy(dynamic_net)
	if setting is not None:
		forward_model.module.set_active_subnet(**setting)
	module_str = forward_model.module.module_str
	# del forward_model
	return module_str

## Used when have to you stop training 
def load_model(
	save_path,
	model,
	optimizer,
	args,
	model_fname=None
):
    latest_fname = os.path.join(save_path, 'latest.txt')
    if model_fname is None and os.path.exists(latest_fname):
        with open(latest_fname, 'r') as fin:
            model_fname = fin.readline()
            if model_fname[-1] == '\n':
                model_fname = model_fname[:-1]
    # noinspection PyBroadException
    try:
        if model_fname is None or not os.path.exists(model_fname):
            model_fname = '%s/checkpoint.pth.tar' % save_path
            with open(latest_fname, 'w') as fout:
                fout.write(model_fname + '\n')
        print("=> loading checkpoint '{}'".format(model_fname))
        checkpoint = torch.load(model_fname, map_location='cpu')
    except Exception:
        print('fail to load checkpoint from %s' % save_path)
        return {}

    model.load_state_dict(checkpoint['state_dict'])
    if 'epoch' in checkpoint:
        args.start_epoch = checkpoint['epoch'] + 1
    if 'best_acc' in checkpoint:
        args.best_acc = checkpoint['best_acc']
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print("=> loaded checkpoint '{}'".format(model_fname))
    return checkpoint


def save_model(
	save_path,
	net,
	optimizer,
	run_config,
	best_acc	= None,
	checkpoint	= None, 
	is_best		= False, 
	model_name	= None
):
	if checkpoint is None:
		checkpoint = {'state_dict': net.state_dict()}
	if model_name is None:
		model_name = 'checkpoint.pth.tar'

	# add `dataset` info to the checkpoint
	checkpoint['dataset'] 	= run_config.dataset  
	checkpoint['optimizer'] = optimizer
	checkpoint['best_acc'] 	= best_acc
	
	latest_fname 	= os.path.join(save_path, 'latest.txt')
	model_path 		= os.path.join(save_path, model_name)
	with open(latest_fname, 'w') as fout:
		fout.write(model_path + '\n')
	torch.save(checkpoint, model_path)

	if is_best:
		best_path = os.path.join(save_path, 'model_best.pth.tar')
		torch.save({'state_dict': checkpoint['state_dict']}, best_path)

def reset_running_statistics(
	net,
	run_config,
	subset_size=2000, 		## changed!!
	####### CHANGED #######s
	subset_batch_size=200, # ## changed!!
	#######################
	distributed = False,
	data_loader=None
):
	if data_loader is None:
		data_loader = run_config.random_sub_train_loader(
			subset_size, 
			subset_batch_size, 
			num_replicas=None, 
			rank=None
		)
	
	set_running_statistics(net, data_loader, distributed) ## FIX ME !!!

def observe_activation(dynamic_net, run_config, args):
	run_config.data_provider.assign_active_img_size(224)
	data_loader = run_config.random_sub_train_loader(
		2000, 
		200, 
		num_replicas=None, 
		rank=None
	)
	set_activation(dynamic_net, data_loader, False)

def train_initialize(model, run_config, args):
	run_config.data_provider.assign_active_img_size(224)
	data_loader = run_config.random_sub_train_loader(
		20000, 
		176, 
		# num_replicas=len(args.gpus.split(',')), 
		# rank=torch.distributed.get_rank(),
		num_replicas=None, 
		rank=None
	)
	train_activation(model, data_loader, False)
