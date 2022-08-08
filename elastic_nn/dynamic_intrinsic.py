import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from int_quantization.fake_quantize import get_fake_quant, LsqQuan
from utils import get_same_padding, sub_filter_start_end, make_divisible, build_activation
from .dynamic_op import DynamicSE
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm

__all__ = [
    'DynamicSepConvBn2DNonFused',
    'DynamicConvBn2DNonFused',
    'DynamicSeparableConvBn2dQuant', 
    'DynamicConvBn2dQuant', 
    # 'DynamicSEQuant', 
]

def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):
    exponential_average_factor = 0.0
    if bn.training and bn.track_running_stats:
        if bn.num_batches_tracked is not None:
            bn.num_batches_tracked += 1
            if bn.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = bn.momentum
    need_sync = bn.training
    if need_sync:
        process_group = torch.distributed.group.WORLD
        world_size = torch.distributed.get_world_size(process_group)
        need_sync = world_size > 1
    
    if not need_sync:
        return F.batch_norm(
            x, 
            bn.running_mean[:feature_dim], 
            bn.running_var[:feature_dim], 
            bn.weight[:feature_dim],
            bn.bias[:feature_dim], 
            training=bn.training,
            momentum=exponential_average_factor, 
            eps=bn.eps,
        )
    else:
        return sync_batch_norm.apply(
            x,
            bn.weight[:feature_dim],
            bn.bias[:feature_dim], 
            bn.running_mean[:feature_dim], 
            bn.running_var[:feature_dim], 
            bn.eps,
            exponential_average_factor,
            process_group,
            world_size,
        )

class DynamicSepConvBn2DNonFused(nn.Module):
    KERNEL_TRANSFORM_MODE = 1  # None or 1

    def __init__(
        self,
        max_in_channels,
        kernel_size_list,
        weight_quant_list='SD4_per_channel',
        act_quant_list = 'int8',
        stride=1, dilation=1,
        eps=1e-5, momentum=0.1,
    ):
        super(DynamicSepConvBn2DNonFused, self).__init__()

        self.max_in_channels   = max_in_channels
        self.kernel_size_list  = kernel_size_list
        self.stride            = stride
        self.dilation          = dilation   

        self.weight_quant_mapping = {}
        self.weight_quant_list = []
        for wq in weight_quant_list:
            if wq == 'fp32':
                self.weight_quant_mapping[wq] = None
            elif 'per_channel' in wq:
                self.weight_quant_mapping[wq] = len(self.weight_quant_list)
                self.weight_quant_list.append(get_fake_quant(wq, max_in_channels))
            else:
                self.weight_quant_mapping[wq] = len(self.weight_quant_list)
                self.weight_quant_list.append(get_fake_quant(wq))
        self.weight_quant_list = nn.ModuleList(self.weight_quant_list)
        
        self.act_quant_mapping = {}
        self.act_quant_list = []
        for aq in act_quant_list:
            if aq == 'fp32':
                self.act_quant_mapping[aq] = None
            else:
                self.act_quant_mapping[aq] = len(self.act_quant_list)
                self.act_quant_list.append(get_fake_quant(aq))
        self.act_quant_list = nn.ModuleList(self.act_quant_list)
        
        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_in_channels,
            max(self.kernel_size_list),
            self.stride,
            groups=self.max_in_channels,
            bias=False
        )
        self.bn = nn.BatchNorm2d(
            self.max_in_channels,
            eps=eps,
            momentum=momentum,
            affine=True,
            track_running_stats=True
        )
        
        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = '%dto%d' % (ks_larger, ks_small)
                # noinspection PyArgumentList
                scale_params['%s_matrix' % param_name] = Parameter(torch.eye(ks_small ** 2))
            for name, param in scale_params.items():
                self.register_parameter(name, param)
        
        self.active_kernel_size = max(self.kernel_size_list)
        self.active_weight_quant = weight_quant_list[0]
        self.active_act_quant = act_quant_list[0]

    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[:out_channel, :in_channel, :, :]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                ######################## KERNEL TRANSFORM ######################
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                	_input_filter, self.__getattr__('%dto%d_matrix' % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks ** 2)
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks, target_ks)
                ################################################################
                start_filter = _input_filter
            filters = start_filter
        return filters
    
    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = out_channel = x.size(1)
        ## Convolution Kernel
        filters = self.get_active_filter(in_channel, kernel_size).contiguous()
        weight_fake_quant = self.weight_quant_list[self.weight_quant_mapping[self.active_weight_quant]] if self.weight_quant_mapping[self.active_weight_quant] is not None else None
        act_fake_quant = self.act_quant_list[self.act_quant_mapping[self.active_act_quant]] if self.act_quant_mapping[self.active_act_quant] is not None else None 
        padding = get_same_padding(kernel_size)
        conv_weight = weight_fake_quant(filters) if weight_fake_quant is not None else filters
        output = F.conv2d(x, conv_weight, None, self.stride, padding, self.dilation, in_channel)
        if act_fake_quant is not None:
            output = act_fake_quant(output)
        output = bn_forward(output, self.bn, out_channel)
        return output

class DynamicConvBn2DNonFused(nn.Module):
    def __init__(
		self, 
		max_in_channels, 
		max_out_channels, 
        weight_quant_list='SD4_per_channel',
        act_quant_list = 'int8',
		kernel_size=1, stride=1, dilation=1,
        eps=1e-5, momentum=0.1
	):
        super(DynamicConvBn2DNonFused, self).__init__()

        self.max_in_channels 	= max_in_channels
        self.max_out_channels 	= max_out_channels
        self.kernel_size 		= kernel_size	# 1
        self.stride 			= stride		# 1
        self.dilation 			= dilation		# 1
        
        self.weight_quant_mapping = {}
        self.weight_quant_list = []
        for wq in weight_quant_list:
            if wq == 'fp32':
                self.weight_quant_mapping[wq] = None
            elif 'per_channel' in wq:
                self.weight_quant_mapping[wq] = len(self.weight_quant_list)
                self.weight_quant_list.append(get_fake_quant(wq, max_out_channels))
            else:
                self.weight_quant_mapping[wq] = len(self.weight_quant_list)
                self.weight_quant_list.append(get_fake_quant(wq))
        self.weight_quant_list = nn.ModuleList(self.weight_quant_list)
        
        self.act_quant_mapping = {}
        self.act_quant_list = []
        for aq in act_quant_list:
            if aq == 'fp32':
                self.act_quant_mapping[aq] = None
            else:
                self.act_quant_mapping[aq] = len(self.act_quant_list)
                self.act_quant_list.append(get_fake_quant(aq))
        self.act_quant_list = nn.ModuleList(self.act_quant_list)
        
        self.conv = nn.Conv2d(
        	self.max_in_channels, 
        	self.max_out_channels, 
        	self.kernel_size, 
        	stride=self.stride, 
        	bias=False,
        )
        self.bn = nn.BatchNorm2d(
            self.max_out_channels,
            eps=eps,
            momentum=momentum,
            affine=True,
            track_running_stats=True
        )
        
        self.active_out_channel = self.max_out_channels
        self.active_weight_quant = weight_quant_list[0]
        self.active_act_quant = act_quant_list[0]
    
    def get_active_filter(self, out_channel, in_channel):
        return self.conv.weight[:out_channel, :in_channel, :, :]

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.get_active_filter(out_channel, in_channel).contiguous()
        weight_fake_quant = self.weight_quant_list[self.weight_quant_mapping[self.active_weight_quant]] if self.weight_quant_mapping[self.active_weight_quant] is not None else None
        act_fake_quant = self.act_quant_list[self.act_quant_mapping[self.active_act_quant]] if self.act_quant_mapping[self.active_act_quant] is not None else None 
        padding = get_same_padding(self.kernel_size)
        conv_weight = weight_fake_quant(filters) if weight_fake_quant is not None else filters
        output = F.conv2d(x, conv_weight, None, self.stride, padding, self.dilation, 1)
        if act_fake_quant is not None:
            output = act_fake_quant(output)
        output = bn_forward(output, self.bn, out_channel)
        return output

    def forward_fused(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.get_active_filter(out_channel, in_channel).contiguous()
        weight_fake_quant = self.weight_quant_list[self.weight_quant_mapping[self.active_weight_quant]] if self.weight_quant_mapping[self.active_weight_quant] is not None else None
        act_fake_quant = self.act_quant_list[self.act_quant_mapping[self.active_act_quant]] if self.act_quant_mapping[self.active_act_quant] is not None else None 
        padding = get_same_padding(self.kernel_size)
        ## BatchNorm
        y_no_grad = F.conv2d(
            x,
            filters,
            None,
            self.stride,
            padding,
            self.dilation,
            1
        )
        ## Calculate batch statistics
        batch_mean = y_no_grad.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
        batch_var = (y_no_grad - batch_mean) * (y_no_grad - batch_mean)
        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        batch_mean = torch.squeeze(batch_mean)
        batch_var = torch.squeeze(batch_var)
        ## Update running mean and variance
        y_no_grad = bn_forward(y_no_grad, self.bn, out_channel)
        ## (Currently Using Simplified Fused version : Only Batch Statistics)
        w_conv = filters.view(out_channel, -1)
        #print(self.bn.weight[:out_channel] / torch.sqrt(batch_var))
        w_bn = torch.diag(self.bn.weight[:out_channel] / torch.sqrt(batch_var))
        conv_weight = torch.mm(w_bn, w_conv).view(filters.size())
        conv_weight = weight_fake_quant(conv_weight) if weight_fake_quant is not None else conv_weight
        conv_bias = self.bn.bias[:out_channel] - self.bn.weight[:out_channel] * batch_mean / torch.sqrt(batch_var)
        ## Fused Convolution
        output = F.conv2d(
        	x, 
            conv_weight, 
            conv_bias, 
            self.stride, 
            padding, 
            self.dilation, 
            1
        )
        if act_fake_quant is not None:
            output = act_fake_quant(output)
        if self.act is not None:
            output = self.act(output)
        return output

class DynamicSeparableConvBn2dQuant(nn.Module):
    KERNEL_TRANSFORM_MODE = 1  # None or 1

    def __init__(
        self, 
        max_in_channels, 
        kernel_size_list, 
        weight_quant_list='SD4_per_channel',
        act_quant_list = 'int8',
        stride=1, 
        dilation=1,
        eps=1e-5, momentum=0.1,
        track_bn_stat=False,
	):
        super(DynamicSeparableConvBn2dQuant, self).__init__()

        self.max_in_channels 	= max_in_channels
        self.kernel_size_list 	= kernel_size_list
        self.stride 			= stride
        self.dilation 			= dilation

        self.conv = nn.Conv2d(
            self.max_in_channels, 
            self.max_in_channels, 
            max(self.kernel_size_list), 
            self.stride,
            groups=self.max_in_channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(
            self.max_in_channels,
            eps=eps,
            momentum=momentum,
            affine=True,
            track_running_stats=True
        )
        self.observer = nn.Sequential()

        self.weight_quant_mapping = {}
        self.weight_quant_list = []
        for wq in weight_quant_list:
            self.weight_quant_mapping[f"fp32"] = None
            if wq != 'fp32':
                if 'per_channel' in wq:
                    self.weight_quant_mapping[wq] = len(self.weight_quant_list)
                    self.weight_quant_list.append(get_fake_quant(wq, max_in_channels))
                else:
                    self.weight_quant_mapping[wq] = len(self.weight_quant_list)
                    self.weight_quant_list.append(get_fake_quant(wq))
        self.weight_quant_list = nn.ModuleList(self.weight_quant_list)
        
        self.act_quant_mapping = {}
        self.act_quant_list = []
        for aq in act_quant_list:
            self.act_quant_mapping['fp32'] = None
            if aq != 'fp32':
                self.act_quant_mapping[aq] = len(self.act_quant_list)
                self.act_quant_list.append(get_fake_quant(aq))
                
        self.act_quant_list = nn.ModuleList(self.act_quant_list)

        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            # register scaling parameters
            # 7to5_matrix, 5to3_matrix
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = '%dto%d' % (ks_larger, ks_small)
                # noinspection PyArgumentList
                scale_params['%s_matrix' % param_name] = Parameter(torch.eye(ks_small ** 2))
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        self.track_bn_stat = track_bn_stat
        self.active_kernel_size = max(self.kernel_size_list)
        self.active_weight_quant = weight_quant_list[0]
        self.active_act_quant = act_quant_list[0]

    def init_lsq(self):
        for name, wquantID in self.weight_quant_mapping.items():
            if 'lsq' in name:
                self.weight_quant_list[wquantID].init_weight([(kernel_size, self.get_active_filter(self.max_in_channels, kernel_size).contiguous()) for kernel_size in reversed(self._ks_set)])

    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[:out_channel, :in_channel, :, :]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                ######################## KERNEL TRANSFORM ######################
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                	_input_filter, self.__getattr__('%dto%d_matrix' % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks ** 2)
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks, target_ks)
                ################################################################
                start_filter = _input_filter
            filters = start_filter
        return filters
    
    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)
        filters = self.get_active_filter(in_channel, kernel_size).contiguous()
        weight_fake_quant = self.weight_quant_list[self.weight_quant_mapping[self.active_weight_quant]] if self.weight_quant_mapping[self.active_weight_quant] is not None else None 
        act_fake_quant = self.act_quant_list[self.act_quant_mapping[self.active_act_quant]] if self.act_quant_mapping[self.active_act_quant] is not None else None 
        padding = get_same_padding(kernel_size)
        
        self.observer(x.detach())
        
        # Quant Activation
        if act_fake_quant is not None:
            x = act_fake_quant(x) 
        
        assert self.bn.running_var is not None
        if weight_fake_quant is not None:
            filters = weight_fake_quant(filters, kernel_size)
        conv = F.conv2d(
        	x, 
            filters, 
            None,#zero_bias, 
            self.stride, 
            padding, 
            self.dilation, 
            in_channel
        )
        if self.track_bn_stat:
            output = self.bn(conv)
        else:
            output = bn_forward(conv, self.bn, in_channel)
        return output

class DynamicConvBn2dQuant(nn.Module):

    def __init__(
    	self, 
    	max_in_channels, 
    	max_out_channels, 
        weight_quant_list='SD4_per_channel',
        act_quant_list = 'int8',
    	kernel_size=1, 
    	stride=1, 
    	dilation=1,
        eps=1e-5, momentum=0.1,
        track_bn_stat=False,
    ):
        super(DynamicConvBn2dQuant, self).__init__()

        self.max_in_channels 	= max_in_channels
        self.max_out_channels 	= max_out_channels
        self.kernel_size 		= kernel_size	# 1
        self.stride 			= stride		# 1
        self.dilation 			= dilation		# 1

        self.weight_quant_mapping = {}
        self.weight_quant_list = []
        for wq in weight_quant_list:
            self.weight_quant_mapping['fp32'] = None
            if wq != 'fp32':        
                if 'per_channel' in wq:
                    self.weight_quant_mapping[wq] = len(self.weight_quant_list)
                    self.weight_quant_list.append(get_fake_quant(wq, max_out_channels))
                else:
                    self.weight_quant_mapping[wq] = len(self.weight_quant_list)
                    self.weight_quant_list.append(get_fake_quant(wq))
        self.weight_quant_list = nn.ModuleList(self.weight_quant_list)

        self.act_quant_mapping = {}
        self.act_quant_list = []
        for aq in act_quant_list:
            self.act_quant_mapping['fp32'] = None
            if aq != 'fp32':
                self.act_quant_mapping[aq] = len(self.act_quant_list)
                self.act_quant_list.append(get_fake_quant(aq))
        self.act_quant_list = nn.ModuleList(self.act_quant_list)

        self.conv = nn.Conv2d(
        	self.max_in_channels, 
        	self.max_out_channels, 
        	self.kernel_size, 
        	stride=self.stride, 
        	bias=False,
        )
        self.bn = nn.BatchNorm2d(
            self.max_out_channels,
            eps=eps,
            momentum=momentum,
            affine=True,
            track_running_stats=True
        )
        self.observer = nn.Sequential()
        self.track_bn_stat = track_bn_stat
        
        self.active_out_channel = self.max_out_channels
        self.active_weight_quant = weight_quant_list[0]
        self.active_act_quant = act_quant_list[0]

    def init_lsq(self):
        for name, wquantID in self.weight_quant_mapping.items():
            if 'lsq' in name:
                self.weight_quant_list[wquantID].init_weight([(self.kernel_size, self.conv.weight)])

    def get_active_filter(self, out_channel, in_channel):
        return self.conv.weight[:out_channel, :in_channel, :, :]

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.get_active_filter(out_channel, in_channel).contiguous()
        weight_fake_quant = self.weight_quant_list[self.weight_quant_mapping[self.active_weight_quant]] if self.weight_quant_mapping[self.active_weight_quant] is not None else None
        act_fake_quant = self.act_quant_list[self.act_quant_mapping[self.active_act_quant]] if self.act_quant_mapping[self.active_act_quant] is not None else None 
        padding = get_same_padding(self.kernel_size)
        
        self.observer(x.detach())
        
        if act_fake_quant is not None:
            x = act_fake_quant(x)
        
        assert self.bn.running_var is not None
        if weight_fake_quant is not None:
            filters = weight_fake_quant(filters, self.kernel_size)
        conv = F.conv2d(
        	x, 
            filters, 
            None,# zero_bias, 
            self.stride, 
            padding, 
            self.dilation, 
            1
        )
        if self.track_bn_stat:
            output = self.bn(conv)
        else:
            output = bn_forward(conv, self.bn, out_channel)
        return output

