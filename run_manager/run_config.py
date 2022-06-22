# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

from utils import calc_learning_rate
from data_providers import ImagenetDataProvider

__all__ = ['RunConfig', 'ImagenetRunConfig', 'DistributedImageNetRunConfig']


class RunConfig:

	def __init__(self, n_epochs, init_lr, lr_schedule_type,
	             dataset, train_batch_size, test_batch_size, train_size, valid_size,
	             label_smoothing, no_decay_keys, mixup_alpha, model_init, 
				 validation_frequency, print_frequency):
		self.n_epochs 			= n_epochs			# 150
		self.init_lr 			= init_lr			# 0.05
		self.lr_schedule_type 	= lr_schedule_type  # 'cosine'

		self.dataset 			= dataset			# 'imagenet'
		self.train_batch_size 	= train_batch_size  # 64
		self.test_batch_size 	= test_batch_size	# 64	
		
		self.train_size         = train_size		# None
		self.valid_size 		= valid_size		# None

		self.label_smoothing 	= label_smoothing	# 0.1
		self.no_decay_keys 		= no_decay_keys		# None

		self.mixup_alpha 		= mixup_alpha		# None

		self.model_init 		= model_init		# 'he_fout'
		self.validation_frequency = validation_frequency # 1
		self.print_frequency 	= print_frequency	# 10

	@property
	def config(self):
		config = {}
		for key in self.__dict__:
			if not key.startswith('_'):
				config[key] = self.__dict__[key]
		return config

	def copy(self):
		return RunConfig(**self.config)

	""" learning rate """

	def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
		""" adjust learning of a given optimizer and return the new learning rate """
		new_lr = calc_learning_rate(epoch, self.init_lr, self.n_epochs, batch, nBatch, self.lr_schedule_type)
		for param_group in optimizer.param_groups:
			param_group['lr'] = new_lr
		return new_lr

	def warmup_adjust_learning_rate(self, optimizer, T_total, nBatch, epoch, batch=0, warmup_lr=0):
		T_cur = epoch * nBatch + batch + 1
		new_lr = T_cur / T_total * (self.init_lr - warmup_lr) + warmup_lr
		for param_group in optimizer.param_groups:
			param_group['lr'] = new_lr
		return new_lr
	
	""" data provider """

	@property
	def data_provider(self):
		raise NotImplementedError

	@property
	def train_loader(self):
		return self.data_provider.train

	@property
	def valid_loader(self):
		return self.data_provider.valid

	@property
	def test_loader(self):
		return self.data_provider.test

	def random_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
		return self.data_provider.build_sub_train_loader(n_images, batch_size, num_worker, num_replicas, rank)

class ImagenetRunConfig(RunConfig):

	def __init__(
		self, 
		n_epochs			= 150, 
		init_lr				= 0.05, 
		lr_schedule_type	= 'cosine',
	    dataset				= 'imagenet', 
		train_batch_size	= 256, 
		test_batch_size		= 500, 
		train_size			= None,
		valid_size			= None,
		label_smoothing		= 0.1, 
		no_decay_keys		= None,
	    mixup_alpha			= None, 
		model_init			= 'he_fout', 
		validation_frequency= 1, 
		print_frequency		= 10,
	    n_worker			= 32, 
		resize_scale		= 0.08, 
		distort_color		= 'tf', 
		image_size			= 224, 
		save_path			= None,
	):
		super(ImagenetRunConfig, self).__init__(
			n_epochs, init_lr, lr_schedule_type,
			dataset, train_batch_size, test_batch_size, train_size, valid_size,
			label_smoothing, no_decay_keys,
			mixup_alpha,
			model_init, validation_frequency, print_frequency
		)
		## Additional Parameters for Data Provider ##
		self.n_worker 		= n_worker				# 32
		self.resize_scale 	= resize_scale			# 0.08
		self.distort_color 	= distort_color			# 'tf'
		self.image_size 	= image_size			# 224
		self.save_path      = save_path
		#############################################

	@property
	def data_provider(self):
		if self.__dict__.get('_data_provider', None) is None:
			if self.dataset == ImagenetDataProvider.name():
				DataProviderClass = ImagenetDataProvider
			else:
				raise NotImplementedError
			self.__dict__['_data_provider'] = DataProviderClass(
				save_path           = self.save_path,
				train_batch_size	= self.train_batch_size, 
				test_batch_size		= self.test_batch_size,
				train_size          = self.train_size,			
				valid_size			= self.valid_size,  ## None
				
				n_worker			= self.n_worker, 
				resize_scale		= self.resize_scale,
				distort_color		= self.distort_color, 
				image_size			= self.image_size,
			)
		return self.__dict__['_data_provider']

class DistributedImageNetRunConfig(ImagenetRunConfig):

	def __init__(
		self, 
		n_epochs			= 150, 
		init_lr				= 0.05, 
		lr_schedule_type	= 'cosine', 
		lr_schedule_param	= None,
	    dataset				= 'imagenet', 
		train_batch_size	= 64, 
		test_batch_size		= 64, 
		train_size			= None,
		valid_size			= None,
	    opt_type			= 'sgd', 
		opt_param			= None, 
		weight_decay		= 4e-5, 
		label_smoothing		= 0.1, 
		no_decay_keys		= None,
	    mixup_alpha			= None, 
		model_init			= 'he_fout', 
		validation_frequency= 1, 
		print_frequency		= 10,
	    n_worker			= 8, 
		resize_scale		= 0.08, 
		distort_color		= 'tf', 
		image_size			= 224,
	    num_replicas		= 1,
		rank				= 0,
		save_path 			= None
	):
		super(DistributedImageNetRunConfig, self).__init__(
			n_epochs, 
			init_lr, 
			lr_schedule_type, 
			dataset, 
			train_batch_size, 
			test_batch_size, 
			train_size,
			valid_size,
			label_smoothing, 
			no_decay_keys,
			mixup_alpha, 
			model_init, 
			validation_frequency, 
			print_frequency, 
			n_worker, 
			resize_scale, 
			distort_color,
			image_size,
			save_path
		)

		############# Newly Add Features ############
		self._num_replicas = num_replicas			#
		self._rank 		   = rank					#
		#############################################

	@property
	def data_provider(self):
		if self.__dict__.get('_data_provider', None) is None:
			if self.dataset == ImagenetDataProvider.name():
				DataProviderClass = ImagenetDataProvider
			else:
				raise NotImplementedError
			self.__dict__['_data_provider'] = DataProviderClass(
				save_path           = self.save_path,	
				train_batch_size	= self.train_batch_size, 
				test_batch_size		= self.test_batch_size,
				train_size          = self.train_size,
				valid_size			= self.valid_size,  ## None
				
				n_worker			= self.n_worker, 
				resize_scale		= self.resize_scale,
				distort_color		= self.distort_color, 
				image_size			= self.image_size,
				
				num_replicas		= self._num_replicas, 
				rank				= self._rank,
			)
		return self.__dict__['_data_provider']
