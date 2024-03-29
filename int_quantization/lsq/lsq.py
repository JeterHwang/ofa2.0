from xxlimited import new
import torch as t
import torch.nn.functional as F
from .quantizer import Quantizer


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True, dynamic=True):
        super().__init__(bit)

        self.bitwidth = bit
        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            self.thd_neg = - 2 ** (bit - 1)
            self.thd_pos = 2 ** (bit - 1) - 1
        
        self.dynamic = dynamic
        self.per_channel = per_channel
        self.s = t.nn.Parameter(t.ones(1))
        self.offset = t.nn.Parameter(t.zeros(1))
        self.kernel_size = []

        if not self.per_channel:
            self.observer = EMA()

    def init_weight(self, weight_list):
        assert self.per_channel and len(weight_list) >= 0

        kernel_large, x = weight_list[0]
        mean = x.detach().mean(dim=list(range(1, x.dim())), keepdim=True)
        std = t.sqrt(((x.detach() - mean) * (x.detach() - mean)).mean(dim=list(range(1, x.dim())), keepdim=True))
        prev_step = 6 * std / (self.thd_pos - self.thd_neg)
        self.s = t.nn.Parameter(prev_step)
        self.kernel_size.append(kernel_large)

        if len(weight_list) > 1:
            for kernel_small, x in weight_list[1:]:
                mean = x.detach().mean(dim=list(range(1, x.dim())), keepdim=True)
                std = t.sqrt(((x.detach() - mean) * (x.detach() - mean)).mean(dim=list(range(1, x.dim())), keepdim=True))
                new_step = 6 * std / (self.thd_pos - self.thd_neg)
                ratio = t.div(new_step, prev_step)
                self.register_parameter(f"{kernel_large}_{kernel_small}_coefficient", t.nn.Parameter(ratio))
                self.kernel_size.append(kernel_small)
                kernel_large = kernel_small
                prev_step = new_step
        # self.s = t.nn.Parameter(x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        # self.offset = t.nn.Parameter(t.zeros_like(self.s))
            
    def init_act(self, Xmin, Xmax):
        assert not self.per_channel
        self.s = t.nn.Parameter((Xmax - Xmin) / (self.thd_pos - self.thd_neg))
        self.offset = t.nn.Parameter(Xmin - self.thd_neg * (Xmax - Xmin) / (self.thd_pos - self.thd_neg))
    
    def get_active_scale(self, kernel_size=None, channel_size=None):
        if self.per_channel:
            step = self.s[:channel_size]
            if kernel_size is not None:
                prev_ks = self.kernel_size[0]
                for ks in self.kernel_size[1:]:
                    if prev_ks == kernel_size:
                        break
                    step = t.mul(step, self.__getattr__(f"{prev_ks}_{ks}_coefficient")[:channel_size])
                    prev_ks = ks
        else:
            xmax, xmin = self.observer()
            delta = (xmax - xmin) / (self.thd_pos - self.thd_neg)
            step = self.s * delta
        return step

    def get_active_offset(self):
        if self.per_channel:
            Beta = self.offset
        else:
            _, xmin = self.observer()
            Beta = xmin + self.offset
        return Beta

    def forward(self, x, kernel_size=None):
        if not self.dynamic:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            s_scale = grad_scale(self.s, s_grad_scale)
            if self.per_channel:
                x = x / s_scale
            else:
                x = (x - self.offset) / s_scale
            x = t.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
            if self.per_channel:
                x = x * s_scale
            else:
                x = x * s_scale + self.offset
            return x
        if self.per_channel:
            channel_size = x.size(0)
            
            step = self.get_active_scale(kernel_size, channel_size)
            
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            s_scale = grad_scale(step, s_grad_scale)

            x = x / s_scale
            x = t.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
            x = x * s_scale
        else:
            channel_size = x.size(1)
            xmax, xmin = self.observer(x)
            delta = (xmax - xmin) / (self.thd_pos - self.thd_neg)
            z = xmin
            
            step = self.s * delta
            Beta = self.offset + z
            
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
            s_scale = grad_scale(step, s_grad_scale)
            
            x = (x - Beta) / s_scale
            x = t.clamp(x, self.thd_neg, self.thd_pos)
            x = round_pass(x)
            x = x * s_scale + Beta
        
        return x

class EMA(t.nn.Module):
    def __init__(self, averaging_constant=0.1):
        super(EMA, self).__init__()
        self.averaging_constant = averaging_constant
        self.register_buffer('min_val', t.tensor([]))
        self.register_buffer('max_val', t.tensor([]))
    
    def forward(self, x=None):
        if self.training and x is not None:
            xmax = t.max(x.detach(), dim=1)[0].mean()
            xmin = t.min(x.detach(), dim=1)[0].mean()
        else:
            xmax = self.max_val
            xmin = self.min_val
        return xmax, xmin