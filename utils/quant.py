import torch
import torch.nn as nn
from int_quantization.fake_quantize import FakeQuantize
from elastic_nn.dynamic_intrinsic import DynamicConvBn2dQuant, DynamicSeparableConvBn2dQuant

__all__ = [
	'profile_quant',
    'disable_bn_stats',
    'enable_bn_stats',
    'is_pareto_efficient',
    'sync_bn',
    'sync_fake_quant',
]
multiply_adds = 1

def count_convNd(m, _, y):
    cin = m.in_channels

    kernel_ops = m.weight.size()[2] * m.weight.size()[3]
    ops_per_element = kernel_ops
    output_elements = y.nelement()

    # cout x oW x oH
    total_ops = cin * output_elements * ops_per_element // m.groups
    m.total_ops = torch.Tensor([int(total_ops)])


def count_linear(m, _, __):
    total_ops = m.in_features * m.out_features

    m.total_ops = torch.Tensor([int(total_ops)])

def count_fused_sep(m, _, y):
    cin = m.conv.in_channels

    kernel_ops = m.active_kernel_size ** 2
    ops_per_element = kernel_ops
    output_elements = y.nelement()

    total_ops = cin * output_elements * ops_per_element // m.conv.groups
    m.total_ops = torch.Tensor([int(total_ops)])

def count_fused_inv(m, _, y):
    cin = m.conv.in_channels

    kernel_ops = m.conv.weight.size()[2] * m.conv.weight.size()[3]
    ops_per_element = kernel_ops
    output_elements = y.nelement()

    total_ops = cin * output_elements * ops_per_element // m.conv.groups
    m.total_ops = torch.Tensor([int(total_ops)])

register_hooks = {
    nn.Conv2d: count_convNd,
    ######################################
    nn.Linear: count_linear,
    ######################################
    DynamicSeparableConvBn2dQuant : count_fused_sep,
    DynamicConvBn2dQuant : count_fused_inv
}


def profile(model, input_size, custom_ops=None):
    handler_collection = []
    custom_ops = {} if custom_ops is None else custom_ops
    
    def add_hooks(m_):
        if len(list(m_.children())) > 0:
            return

        m_.register_buffer('total_ops', torch.zeros(1))
        m_.register_buffer('total_params', torch.zeros(1))

        for p in m_.parameters():
            m_.total_params += torch.Tensor([p.numel()])

        m_type = type(m_)
        fn = None

        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
        else:
            # print("Not implemented for ", m_)
            pass

        if fn is not None:
            # print("Register FLOP counter for module %s" % str(m_))
            _handler = m_.register_forward_hook(fn)
            handler_collection.append(_handler)

    original_device = model.parameters().__next__().device
    training = model.training

    model.eval()
    model.apply(add_hooks)

    x = torch.zeros(input_size).to(original_device)
    with torch.no_grad():
        model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        total_ops += m.total_ops
        total_params += m.total_params

    total_ops = total_ops.item()
    total_params = total_params.item()

    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()

    return total_ops, total_params

def profile_quant(model, input_size, custom_ops=None):
    handler_collection = []
    custom_ops = {} if custom_ops is None else custom_ops
    
    def add_hooks(m_):
        if len(list(m_.children())) > 0 and not isinstance(m_, DynamicConvBn2dQuant) and not isinstance(m_, DynamicSeparableConvBn2dQuant):
            return

        m_.register_buffer('total_ops', torch.zeros(1))

        m_type = type(m_)
        fn = None

        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
        else:
            pass

        if fn is not None:
            # print("Register FLOP counter for module %s" % str(m_))
            _handler = m_.register_forward_hook(fn)
            handler_collection.append(_handler)

    original_device = model.parameters().__next__().device
    training = model.training

    model.eval()
    model.apply(add_hooks)
    x = torch.zeros(input_size).to(original_device)

    with torch.no_grad():
        model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0 and not isinstance(m, DynamicConvBn2dQuant) and not isinstance(m, DynamicSeparableConvBn2dQuant):  # skip for non-leaf module
            continue
        if isinstance(m, DynamicConvBn2dQuant) or isinstance(m, DynamicConvBn2dQuant):
            act_quant = m.act_quant_list[m.act_quant_mapping[m.active_act_quant]] if m.act_quant_mapping[m.active_act_quant] is not None else None
            weight_quant = m.weight_quant_list[m.weight_quant_mapping[m.active_weight_quant]] if m.weight_quant_mapping[m.active_weight_quant] is not None else None
            activation_bit = act_quant.activation_post_process.bitwidth if act_quant is not None else 32
            weight_bit = weight_quant.activation_post_process.bitwidth if weight_quant is not None else 32
            total_ops += m.total_ops * activation_bit * weight_bit
            # print('bits for a and w', activation_bit, weight_bit)
            # print('flops', m.total_ops, 'flops_quant', m.total_ops / 64 * activation_bit * weight_bit)
            for p in m.conv.parameters():
                total_params += torch.tensor([p.numel()]) * weight_bit
            for p in m.bn.parameters():
                total_params += torch.tensor([p.numel()]) * 32
        else:
            if m.total_ops != 0: ## The nn,Conv2d, nn.Batchnorm2d must not be counted
                total_ops += m.total_ops
                for p in m.parameters():
                    total_params += torch.tensor([p.numel()]) * 32
    
    total_ops = total_ops.item()
    total_params = total_params.item()

    model.train(training).to(original_device)
    for handler in handler_collection:
        handler.remove()

    return total_ops / 1e9, total_params / 1e6 / 8

def disable_bn_stats(mod):
	if isinstance(mod, DynamicSeparableConvBn2dQuant) or isinstance(mod, DynamicConvBn2dQuant):
		mod.bn.track_running_stats = False
def enable_bn_stats(mod):
    if isinstance(mod, DynamicSeparableConvBn2dQuant) or isinstance(mod, DynamicConvBn2dQuant):
        mod.bn.track_running_stats = True

def is_pareto_efficient(costs, MAX=True):
	# Sort the list in either ascending or descending order of X
	myList = sorted([(cost[0], cost[1], i) for i, cost in enumerate(costs)], reverse=MAX)
	p_front = [myList[0]]
	p_front_id = [myList[0][2]]    
	# Loop through the sorted list
	for pair in myList[1:]:
		if MAX: 
			if pair[1] >= p_front[-1][1]: # Look for higher values of Y…
				p_front.append(pair) # … and add them to the Pareto frontier
				p_front_id.append(pair[2])
		else:
			if pair[1] <= p_front[-1][1]: # Look for lower values of Y…
				p_front.append(pair) # … and add them to the Pareto frontier
				p_front_id.append(pair[2])
	# Turn resulting pairs back into a list of Xs and Ys
	return p_front_id

def sync_bn(mod):
    if isinstance(mod, DynamicSeparableConvBn2dQuant) or isinstance(mod, DynamicConvBn2dQuant):
        mod.bn.running_mean = mod.bn.running_mean.cuda()
        mod.bn.running_var = mod.bn.running_var.cuda()
        torch.distributed.all_reduce(mod.bn.running_mean, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(mod.bn.running_var, op=torch.distributed.ReduceOp.SUM)
        mod.bn.running_mean /= torch.distributed.get_world_size()
        mod.bn.running_var /= torch.distributed.get_world_size()

def sync_fake_quant(mod):
    if isinstance(mod, FakeQuantize):
        if mod.max_channel is not None:
            min_vals = mod.activation_post_process.min_vals.cuda()
            max_vals = mod.activation_post_process.max_vals.cuda()
            torch.distributed.all_reduce(min_vals, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(max_vals, op=torch.distributed.ReduceOp.SUM)
            min_vals /= torch.distributed.get_world_size()
            max_vals /= torch.distributed.get_world_size()
            mod.activation_post_process.min_vals.data.copy_(min_vals.data)
            mod.activation_post_process.max_vals.data.copy_(max_vals.data)
        else:
            min_val = mod.activation_post_process.min_val.cuda()
            max_val = mod.activation_post_process.max_val.cuda()
            torch.distributed.all_reduce(min_val, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(max_val, op=torch.distributed.ReduceOp.SUM)
            min_val /= torch.distributed.get_world_size()
            max_val /= torch.distributed.get_world_size()
            mod.activation_post_process.min_val.data.copy_(min_val.data)
            mod.activation_post_process.max_val.data.copy_(max_val.data)
        