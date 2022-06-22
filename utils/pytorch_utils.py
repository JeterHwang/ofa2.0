# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import math
import copy
import time
import torch
import torch.nn as nn

__all__ = [
	'mix_images', 
    'mix_labels',
	'label_smooth', 
	'get_net_device',
    'cross_entropy_loss_with_soft_target', 
    'cross_entropy_with_label_smoothing',
	'build_optimizer',
	'calc_learning_rate',
]


""" Mixup """
def mix_images(images, lam):
	flipped_images = torch.flip(images, dims=[0])  # flip along the batch dimension
	return lam * images + (1 - lam) * flipped_images


def mix_labels(target, lam, n_classes, label_smoothing=0.1):
	onehot_target = label_smooth(target, n_classes, label_smoothing)
	flipped_target = torch.flip(onehot_target, dims=[0])
	return lam * onehot_target + (1 - lam) * flipped_target


""" Label smooth """
def label_smooth(target, n_classes: int, label_smoothing=0.1):
	# convert to one-hot
	batch_size = target.size(0)
	target = torch.unsqueeze(target, 1)
	soft_target = torch.zeros((batch_size, n_classes), device=target.device)
	soft_target.scatter_(1, target, 1)
	# label smoothing
	soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
	return soft_target


def cross_entropy_loss_with_soft_target(pred, soft_target):
	logsoftmax = nn.LogSoftmax()
	return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))
	## mean : mean on the batch_size
	## sum  : sum of softmax value on the n_class
	## soft_target 		: (batch_size * n_classes)
	## logsoftmax(pred) : (batch_size * n_classes)

def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
	soft_target = label_smooth(target, pred.size(1), label_smoothing)
	return cross_entropy_loss_with_soft_target(pred, soft_target)

""" optimizer """
def build_optimizer(net_params, opt_type, opt_param, init_lr, weight_decay, no_decay_keys):
	if no_decay_keys is not None:
		assert isinstance(net_params, list) and len(net_params) == 2
		net_params = [
			{'params': net_params[0], 'weight_decay': weight_decay},
			{'params': net_params[1], 'weight_decay': 0},
		]
	else:
		net_params = [{'params': net_params, 'weight_decay': weight_decay}]

	if opt_type == 'sgd':
		opt_param = {} if opt_param is None else opt_param
		momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
		optimizer = torch.optim.SGD(net_params, init_lr, momentum=momentum, nesterov=nesterov)
	elif opt_type == 'adam':
		optimizer = torch.optim.Adam(net_params, init_lr)
	else:
		raise NotImplementedError
	return optimizer

""" Network profiling """
def get_net_device(net):
	return net.parameters().__next__().device



""" learning rate schedule """

def calc_learning_rate(
    epoch, 
	init_lr, 
	n_epochs, 
	batch=0, 
	nBatch=None, 
	lr_schedule_type="cosine",
	lowest_lr=1e-5
):
	if lr_schedule_type == "cosine":
		t_total = n_epochs * nBatch
		t_cur = epoch * nBatch + batch
		lr = 0.5 * (init_lr + lowest_lr) + 0.5 * (init_lr - lowest_lr) * math.cos(math.pi * t_cur / t_total)
	elif lr_schedule_type == 'linear':
		t_total = n_epochs * nBatch
		t_cur = epoch * nBatch + batch
		lr = init_lr - (init_lr - lowest_lr) * t_cur / t_total
	elif lr_schedule_type is None:
		lr = init_lr
	else:
		raise ValueError("do not support: %s" % lr_schedule_type)
	return lr