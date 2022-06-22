import numpy as np
import os
import sys
import torch

try:
	from urllib import urlretrieve
except ImportError:
	from urllib.request import urlretrieve

__all__ = [
    'get_same_padding',
	'sub_filter_start_end',
	'list_sum',
	'list_mean',
	'min_divisible_value',
	'AverageMeter',
	'DistributedMetric',
	'DistributedTensor',
	'val2list',
	'subset_mean',
	'download_url',
	'accuracy',
]

def get_same_padding(kernel_size):
	if isinstance(kernel_size, tuple):
		assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
		p1 = get_same_padding(kernel_size[0])
		p2 = get_same_padding(kernel_size[1])
		return p1, p2
	assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
	assert kernel_size % 2 > 0, 'kernel size should be odd number'
	return kernel_size // 2

def list_sum(x):
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])

def list_mean(x):
    return x[0] if isinstance(x[0], str) else list_sum(x) / len(x)

def sub_filter_start_end(kernel_size, sub_kernel_size):
	center = kernel_size // 2
	dev = sub_kernel_size // 2
	start, end = center - dev, center + dev + 1
	assert end - start == sub_kernel_size
	return start, end

def min_divisible_value(n1, v1):
	""" make sure v1 is divisible by n1, otherwise decrease v1 """
	if v1 >= n1:
		return n1
	while n1 % v1 != 0:
		v1 -= 1
	return v1

def val2list(val, repeat_time=1):
	if isinstance(val, list) or isinstance(val, np.ndarray):
		return val
	elif isinstance(val, tuple):
		return list(val)
	else:
		return [val for _ in range(repeat_time)]

def subset_mean(val_list, sub_indexes):
	sub_indexes = val2list(sub_indexes, 1)
	return list_mean([val_list[idx] for idx in sub_indexes])

def download_url(url, model_dir='~/.torch/', overwrite=False):
	target_dir = url.split('/')[-1]
	model_dir = os.path.expanduser(model_dir)
	try:
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)
		model_dir = os.path.join(model_dir, target_dir)
		cached_file = model_dir
		if not os.path.exists(cached_file) or overwrite:
			sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
			urlretrieve(url, cached_file)
		return cached_file
	except Exception as e:
		# remove lock file so download can be executed next time.
		os.remove(os.path.join(model_dir, 'download.lock'))
		sys.stderr.write('Failed to download from url %s' % url + '\n' + str(e) + '\n')
		return None

def accuracy(output, target, topk=(1,)):
	""" Computes the precision@k for the specified values of k """
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.reshape(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res



class AverageMeter(object):
	"""
	Computes and stores the average and current value
	Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""

	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
	

class DistributedMetric(object):
	"""
		Horovod: average metrics from distributed training.
	"""
	def __init__(self, name, world_size):
		self.name = name
		self.sum = torch.zeros(1)[0]
		self.count = torch.zeros(1)[0]
		self.world_size = world_size

	def update(self, val, delta_n=1):
		if not torch.is_tensor(delta_n):
			delta_n = torch.tensor(delta_n).cuda()
		if not torch.is_tensor(val):
			val = torch.tensor(val).cuda()
		val = val * delta_n
		torch.distributed.all_reduce(val, op=torch.distributed.ReduceOp.SUM)
		torch.distributed.all_reduce(delta_n, op=torch.distributed.ReduceOp.SUM)
		self.sum += val.detach().cpu()
		self.count += delta_n.cpu()

	def reset(self):
		self.sum = torch.zeros(1)[0]
		self.count = torch.zeros(1)[0]

	@property
	def avg(self):
		return self.sum / self.count

class DistributedTensor(object):

	def __init__(self, name, world_size):
		self.name = name
		self.sum = None
		self.count = torch.zeros(1)[0]
		self.synced = False
		self.world_size = world_size

	def update(self, val, delta_n=1):
		val *= delta_n
		if self.sum is None:
			self.sum = val.detach()  # No need to map to CPU cause 
		else:						 # set_running_statistics only 
			self.sum += val.detach() # do on CPU
		self.count += delta_n

	@property
	def avg(self):
		if not self.synced:
			torch.distributed.all_reduce(self.sum, op=torch.distributed.ReduceOp.SUM)
			torch.distributed.all_reduce(self.count, op=torch.distributed.ReduceOp.SUM)
			self.synced = True
		return self.sum / self.count
