from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.nn import Module
from .observer import MovingAverageMinMaxObserver, HistogramObserver, MovingAveragePerChannelMinMaxObserver, PercentileObserver, _with_args
from .quant import Log_Preprocess_cpu, Log_Preprocess_gpu
from .lsq import LsqQuan
# import torch.quantization.fake_quantize
class FakeQuantize(Module):
    r""" Simulate the quantize and dequantize operations in training time.
    The output of this module is given by

    x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max)-zero_point)*scale



    * :attr:`scale` defines the scale factor used for quantization.

    * :attr:`zero_point` specifies the quantized value to which 0 in floating point maps to

    * :attr:`quant_min` specifies the minimum allowable quantized value.

    * :attr:`quant_max` specifies the maximum allowable quantized value.

    * :attr:`fake_quant_enable` controls the application of fake quantization on tensors, note that
      statistics can still be updated.

    * :attr:`observer_enable` controls statistics collection on tensors

    * :attr:`dtype` specifies the quantized dtype that is being emulated with fake-quantization,
                    allowable values are torch.qint8 and torch.quint8. The values of quant_min and
                    quant_max should be chosen to be consistent with the dtype


    Args:
        observer (module): Module for observing statistics on input tensors and calculating scale
                           and zero-point.
        quant_min (int): The minimum allowable quantized value.
        quant_max (int): The maximum allowable quantized value.
        observer_kwargs (optional): Arguments for the observer module

    Attributes:
        observer (Module): User provided module that collects statistics on the input tensor and
                           provides a method to calculate scale and zero-point.

    """
    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, **observer_kwargs):
        super(FakeQuantize, self).__init__()
        assert quant_min <= quant_max, \
            'quant_min must be less than or equal to quant_max'
        self.quant_min = quant_min
        self.quant_max = quant_max
        # fake_quant_enabled and observer_enabled are buffers to support their
        # replication in DDP. Data type is uint8 because NCCL does not support
        # bool tensors.
        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('observer_enabled', torch.tensor([1], dtype=torch.uint8))
        self.activation_post_process = observer(**observer_kwargs)
        self.log_scale = self.activation_post_process.log_scale
        #assert torch.iinfo(self.activation_post_process.dtype).min <= quant_min, 'quant_min out of bound'
        #assert quant_max <= torch.iinfo(self.activation_post_process.dtype).max, 'quant_max out of bound'
        self.register_buffer('scale', torch.tensor([1.0]))
        self.register_buffer('zero_point', torch.tensor([0]))
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        if hasattr(self.activation_post_process, 'max_channel'):
            self.max_channel = self.activation_post_process.max_channel
        else:
            self.max_channel = None
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1

    @torch.jit.export
    def enable_fake_quant(self, enabled=True):
        # type: (bool) -> FakeQuantize
        self.fake_quant_enabled[0] = 1 if enabled else 0
        return self

    @torch.jit.export
    def disable_fake_quant(self):
        return self.enable_fake_quant(False)

    @torch.jit.export
    def enable_observer(self, enabled=True):
        # type: (bool) -> FakeQuantize
        #print(enabled)
        self.observer_enabled[0] = 1 if enabled else 0
        return self

    @torch.jit.export
    def disable_observer(self):
        return self.enable_observer(False)

    @torch.jit.export
    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

    @torch.jit.ignore
    def sync_observer(self):
        if (torch.distributed.is_initialized()):
            torch.distributed.broadcast(self.scale, 0)
            torch.distributed.broadcast(self.zero_point, 0)

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            # print('observer enabled')
            self.activation_post_process(X.detach()) # update observer statistics
            #print('finish observer forward')
        _scale, _zero_point = self.calculate_qparams()
        _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
        if self.max_channel is not None and hasattr(self.activation_post_process, 'ch_axis'):
            feature_dim = X.size(self.ch_axis)
            _scale, _zero_point = _scale[:feature_dim], _zero_point[:feature_dim]
        # print(_scale, _zero_point)
        self.scale.resize_(_scale.shape)
        self.scale.copy_(_scale)
        self.zero_point.resize_(_zero_point.shape)
        self.zero_point.copy_(_zero_point)
        
        if torch.distributed.get_rank() == 0 and self.max_channel is not None and self.scale.size(0) != X.size(self.ch_axis):
            print(X.size(self.ch_axis), self.scale.size(0), self.zero_point.size(0))
            raise AssertionError

        if self.fake_quant_enabled[0] == 1:
            if (self.log_scale):
                # X = Log_Preprocess_gpu(X, self.scale)
                X = Log_Preprocess_cpu(X, self.scale)

            if self.qscheme == torch.per_channel_symmetric or self.qscheme == torch.per_channel_affine:
                # print(X.size())
                # print(self.scale.size())
                # print(self.zero_point.size())
                X = torch.fake_quantize_per_channel_affine(X, self.scale, self.zero_point, self.ch_axis, self.quant_min, self.quant_max)
            else:
                # if torch.distributed.get_rank() == 0:
                    #print(self.scale, self.zero_point, self.quant_min, self.quant_max)
                    #print("Before Quantized : ", X)
                X = torch.fake_quantize_per_tensor_affine(X, float(self.scale), int(self.zero_point), self.quant_min, self.quant_max)
                # if torch.distributed.get_rank() == 0:
                    #print("After Quantized : ", X)
        return X

    with_args = classmethod(_with_args)

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={},\
            scale={}, zero_point={}'.format(
            self.fake_quant_enabled, self.observer_enabled,
            self.scale, self.zero_point)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super(FakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(FakeQuantize, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                        missing_keys, unexpected_keys, error_msgs)

default_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=254,
                                            dtype=torch.quint8, qscheme=torch.per_tensor_symmetric, bitwidth=8)

default_int4_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=15,
                                                 dtype=torch.quint8, qscheme=torch.per_tensor_affine, bitwidth=4)
default_int6_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=63,
                                                 dtype=torch.quint8, qscheme=torch.per_tensor_affine, bitwidth=6)
default_int8_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255,
                                                 dtype=torch.quint8, qscheme=torch.per_tensor_affine, bitwidth=8)

default_log4_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=128,
                                                 dtype=torch.quint8, qscheme=torch.per_tensor_symmetric, bitwidth=4, log_scale=True)

default_weight_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=-7, quant_max=7,
                                                   dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, bitwidth=4)

default_per_channel_weight_fake_quant = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                               quant_min=-7,
                                                               quant_max=7,
                                                               dtype=torch.qint8,
                                                               qscheme=torch.per_channel_symmetric,
                                                               bitwidth=4,
                                                               ch_axis=0)

default_histogram_fake_quant = FakeQuantize.with_args(observer=HistogramObserver,
                                                      quant_min=0,
                                                      quant_max=126,
                                                      dtype=torch.quint8,
                                                      qscheme=torch.per_tensor_affine,
                                                      bitwidth=7)

default_percentile_fake_quant = FakeQuantize.with_args(observer=PercentileObserver, quant_min=0, quant_max=254,
                                                       dtype=torch.quint8, qscheme=torch.per_tensor_symmetric, bitwidth=8)

default_int4_percentile_fake_quant = FakeQuantize.with_args(observer=PercentileObserver, quant_min=0, quant_max=14, percentile=0.999,
                                                            dtype=torch.quint8, qscheme=torch.per_tensor_affine, bitwidth=4)

default_log_weight_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=-64, quant_max=64,
                                                       dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, bitwidth=4, log_scale=True)

default_per_channel_log_weight_fake_quant = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                                   quant_min=-64,
                                                                   quant_max=64,
                                                                   dtype=torch.qint8,
                                                                   qscheme=torch.per_channel_symmetric,
                                                                   bitwidth=4,
                                                                   log_scale=True,
                                                                   ch_axis=0)
quant_mapping = {
    # Weight
    'int4_per_tensor' : default_weight_fake_quant,
    'SD4_per_tensor' : default_log_weight_fake_quant,
    # Activation
    'int8' : default_int8_fake_quant,
    'int6' : default_int6_fake_quant,
    'int4' : default_int4_fake_quant,
    'SD4' : default_log4_fake_quant,
}

def disable_fake_quant(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        mod.disable_fake_quant()

def enable_fake_quant(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        mod.enable_fake_quant()

def disable_observer(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        mod.disable_observer()

def enable_observer(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        mod.enable_observer()

def sync_observer(mod):
    if type(mod) in set([FakeQuantize]):
        mod.sync_observer()

def disable_act_fake_quant(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        if (mod.activation_post_process.dtype == torch.quint8):
            mod.disable_fake_quant()

def enable_act_fake_quant(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        if (mod.activation_post_process.dtype == torch.quint8):
            mod.enable_fake_quant()

def disable_act_observer(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        if (mod.activation_post_process.dtype == torch.quint8):
            mod.disable_observer()

def enable_act_observer(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        if (mod.activation_post_process.dtype == torch.quint8):
            mod.enable_observer()

def disable_weight_fake_quant(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        if (mod.activation_post_process.dtype == torch.qint8):
            mod.disable_fake_quant()

def enable_weight_fake_quant(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        if (mod.activation_post_process.dtype == torch.qint8):
            mod.enable_fake_quant()

def disable_weight_observer(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        if (mod.activation_post_process.dtype == torch.qint8):
            mod.disable_observer()

def enable_weight_observer(mod):
    if type(mod) in set([FakeQuantize, torch.quantization.FakeQuantize]):
        if (mod.activation_post_process.dtype == torch.qint8):
            mod.enable_observer()

def get_fake_quant(fake_quant_type, max_channel=None, dynamic=True):
    assert fake_quant_type is not None
    if fake_quant_type in quant_mapping:
        return quant_mapping[fake_quant_type]()
    elif fake_quant_type =='lsq3_per_tensor' : 
        return LsqQuan(3, True, False, False, dynamic)
    elif fake_quant_type =='lsq4_per_tensor' : 
        return LsqQuan(4, True, False, False, dynamic)
    elif fake_quant_type =='lsq5_per_tensor' : 
        return LsqQuan(5, True, False, False, dynamic)
    elif fake_quant_type =='lsq6_per_tensor' : 
        return LsqQuan(6, True, False, False, dynamic)
    elif fake_quant_type =='lsq8_per_tensor' : 
        return LsqQuan(8, True, False, False, dynamic)
    elif fake_quant_type == 'lsq3_per_channel' :
        return LsqQuan(3, False, True, True, dynamic)
    elif fake_quant_type == 'lsq4_per_channel' :
        return LsqQuan(4, False, True, True, dynamic)
    elif max_channel is not None:
        if fake_quant_type == 'int4_per_channel':
            return FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, quant_min=-7, quant_max=7, 
                                            dtype=torch.qint8, qscheme=torch.per_channel_symmetric, bitwidth=4,
                                                ch_axis=0, max_channel=max_channel)()
        elif fake_quant_type == 'int6_per_channel':
            return FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, quant_min=-31, quant_max=31, 
                                            dtype=torch.qint8, qscheme=torch.per_channel_symmetric, bitwidth=6,
                                                ch_axis=0, max_channel=max_channel)()
        elif fake_quant_type == 'int8_per_channel':
            return FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, quant_min=-127, quant_max=127, 
                                            dtype=torch.qint8, qscheme=torch.per_channel_symmetric, bitwidth=8,
                                                ch_axis=0, max_channel=max_channel)()
        elif fake_quant_type == 'SD3_per_channel':
            return FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, quant_min=-4, quant_max=4,
                                            dtype=torch.qint8, qscheme=torch.per_channel_symmetric, bitwidth=3,
                                                log_scale=True, ch_axis=0, max_channel=max_channel)()
        elif fake_quant_type == 'SD4_per_channel':
            return FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, quant_min=-64, quant_max=64,
                                            dtype=torch.qint8, qscheme=torch.per_channel_symmetric, bitwidth=4,
                                                log_scale=True, ch_axis=0, max_channel=max_channel)()
        elif fake_quant_type == 'SD5_per_channel':
            return FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, quant_min=-16384, quant_max=16384,
                                            dtype=torch.qint8, qscheme=torch.per_channel_symmetric, bitwidth=5,
                                                log_scale=True, ch_axis=0, max_channel=max_channel)()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError