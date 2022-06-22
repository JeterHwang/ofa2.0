# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
import numpy as np
import os
import random
import json

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from elastic_nn.dynamic_intrinsic import DynamicSepConvBn2DNonFused

from training                     import train, validate_all_settings, load_model, load_pretrained_model, train_elastic_depth, train_elastic_expand, train_elastic_bit
from networks                     import OFAMobileNetV3, MobileNetV3Large
from elastic_nn.dynamic_op        import DynamicSeparableConv2d
from elastic_nn.dynamic_intrinsic import DynamicSepConvBn2DNonFused
from run_manager.run_config       import DistributedImageNetRunConfig    ## RunConfig
from utils                        import MyRandomResizedCrop
from utils.common_tools           import download_url 
from utils.my_modules             import init_models
from utils.pytorch_utils          import cross_entropy_loss_with_soft_target, cross_entropy_with_label_smoothing, build_optimizer

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, default='bit', choices=['kernel', 'depth', 'expand', 'bit'])
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--manual_seed', type=int, default=0)

    parser.add_argument('--lr_schedule_type', type=str, default='cosine')
    parser.add_argument('--base_batch_size', type=int, default=64)
    parser.add_argument('--train_size', type=float, default=0.1)
    parser.add_argument('--valid_size', type=float, default=0.05)

    parser.add_argument('--opt_type', type=str, default='sgd')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--no_nesterov', type=bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=3e-5)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--no_decay_keys', type=str, default='bn#bias')
    parser.add_argument('--fp16_allreduce', type=bool, default=False)

    parser.add_argument('--model_init', type=str, default='he_fout')
    parser.add_argument('--validation_frequency', type=int, default=4)
    parser.add_argument('--print_frequency', type=int, default=10)

    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--resize_scale', type=float, default=0.08)
    parser.add_argument('--distort_color', type=str, default='tf')
    parser.add_argument('--image_size', type=str, default='128,160,192,224')
    parser.add_argument('--continuous_size', type=bool, default=True)
    parser.add_argument('--not_sync_distributed_image_size', type=bool, default=False)

    parser.add_argument('--bn_momentum', type=float, default=0.1)
    parser.add_argument('--bn_eps', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--base_stage_width', type=str, default='proxyless')

    parser.add_argument('--width_mult_list', type=str, default='1.0')
    parser.add_argument('--dy_conv_scaling_mode', type=int, default=1)
    parser.add_argument('--independent_distributed_sampling', type=bool, default=False)

    parser.add_argument('--kd_ratio', type=float, default=1.0)
    parser.add_argument('--kd_type', type=str, default='ce')

    parser.add_argument('--hyperparameter_path', type=Path, default='./')
    parser.add_argument("--gpus", type=str, help="gpu devices id", default="0,2,3")
    
    parser.add_argument('--save_path', type=Path, default='/home/ntu329/Documents/Datasets/ImageNet')
    parser.add_argument('--model_fname', type=str, default=None)
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    # parser.add_argument('--epochs',)
    args = parser.parse_args()
    return args

def add_args(args) -> Namespace:
    path2hyperparameter = args.hyperparameter_path / 'hyperparameter.json'
    hyperparameters: Dict[str : Dict] = json.loads(path2hyperparameter.read_text())
    hyperparameters = hyperparameters[args.task][f'phase{args.phase}']
    
    arg_dict = vars(args)
    arg_dict.update(hyperparameters)
    args = Namespace(**arg_dict)

    args.teacher_path = download_url(
        'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K7',
        model_dir='.torch/ofa_checkpoints/0'
    )
    #################### TRAINING PARAMETERS ##################
    args.lr_schedule_param = None
    args.mixup_alpha = None # Originally defined in ImagenetRunconfig
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    args.init_lr = args.base_lr
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr

    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size * 4
    if args.dy_conv_scaling_mode == -1:
        args.dy_conv_scaling_mode = None
    
    ###################### MODEL PARAMETERS ###################
    args.image_size = [int(img_size) for img_size in args.image_size.split(',')]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]
    args.ks_list            = [int(ks) for ks in args.ks_list.split(',')]
    args.expand_list        = [int(e) for e in args.expand_list.split(',')]
    args.depth_list         = [int(d) for d in args.depth_list.split(',')]
    args.weight_quant_list  = [wq for wq in args.weight_quant_list.split(',')]
    args.act_quant_list     = [wq for wq in args.act_quant_list.split(',')]
    args.width_mult_list    = [float(width_mult) for width_mult in args.width_mult_list.split(',')]
    # Must add this line otherwise OFAMobilenetV3 will fail to initiate
    args.width_mult_list    = (
        args.width_mult_list[0]
        if len(args.width_mult_list) == 1
        else args.width_mult_list
    ) 
    return args

def same_seed(seed):
    # set seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

def distributed_training(gpu, args):
    rank = gpu
    dist.init_process_group(
    	backend     = args.backend,
        init_method = 'env://',
    	world_size  = args.world_size,
    	rank        = rank
    )
    
    same_seed(args.manual_seed)
    
    if rank == 0:
        print('----- Finish Setting Seed -----')

    MyRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size
    DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode
    DynamicSepConvBn2DNonFused.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode

    torch.cuda.set_device(gpu)

    ############### Configuration ###################
    # print(args.image_size)
    run_config = DistributedImageNetRunConfig(
        n_epochs                = args.n_epochs, 
        init_lr                 = args.init_lr, 
        lr_schedule_type        = args.lr_schedule_type, 
        lr_schedule_param       = args.lr_schedule_param, 
        train_batch_size        = args.train_batch_size,
		test_batch_size         = args.test_batch_size, 
        train_size              = args.train_size,
        valid_size              = args.valid_size, 
        opt_type                = args.opt_type, 
        opt_param               = args.opt_param, 
        weight_decay            = args.weight_decay, 
        label_smoothing         = args.label_smoothing, 
        no_decay_keys           = args.no_decay_keys, 
        mixup_alpha             = args.mixup_alpha,
		model_init              = args.model_init, 
        validation_frequency    = args.validation_frequency, 
        print_frequency         = args.print_frequency, 
        n_worker                = args.n_worker,
		resize_scale            = args.resize_scale, 
        distort_color           = args.distort_color, 
        image_size              = args.image_size, 
        num_replicas            = args.world_size, 
        save_path               = args.save_path,  
        rank                    = rank
    )
    ################## Model ####################
    net = OFAMobileNetV3(
        n_classes           = run_config.data_provider.n_classes, 
        bn_param            = (args.bn_momentum, args.bn_eps),
        dropout_rate        = args.dropout, 
        base_stage_width    = args.base_stage_width, 
        width_mult          = args.width_mult_list,
        ks_list             = args.ks_list, 
        expand_ratio_list   = args.expand_list, 
        depth_list          = args.depth_list,
        weight_quant_list   = args.weight_quant_list,
		act_quant_list		= args.act_quant_list,
    ).cuda()

    init_models(net, args.model_init)
    
    if rank == 0 : 
        print('Model Initialized !!')
    ############## Teacher Model ################
    if args.kd_ratio > 0:
        args.teacher_model = MobileNetV3Large(
            n_classes       = run_config.data_provider.n_classes, 
            bn_param        = (args.bn_momentum, args.bn_eps),
            dropout_rate    = 0, 
            width_mult      = 1.0, 
            ks              = 7, 
            expand_ratio    = 6, 
            depth_param     = 4,
        ).cuda()

    # load teacher net weights
    if args.kd_ratio > 0:
        load_pretrained_model(
            args.teacher_model, 
            model_path=args.teacher_path
        )
    ############## Criterions ################
    if isinstance(args.mixup_alpha, float):
        train_criterion = cross_entropy_loss_with_soft_target
    elif args.label_smoothing > 0:
        train_criterion = \
            lambda pred, target: cross_entropy_with_label_smoothing(pred, target, args.label_smoothing)
    else:
        train_criterion = nn.CrossEntropyLoss()
        
    # test criterion
    test_criterion = nn.CrossEntropyLoss()
################### Optimizer ###############
    if args.no_decay_keys:
        keys = args.no_decay_keys.split('#')
        net_params = [
            net.get_parameters(keys, mode='exclude'),  # parameters with weight decay
            net.get_parameters(keys, mode='include'),  # parameters without weight decay
        ]
    else:
        # noinspection PyBroadException
        try:
            net_params = net.weight_parameters()
        except Exception:
            net_params = []
            for param in net.parameters():
                if param.requires_grad:
                    net_params.append(param)
    optimizer = build_optimizer(
        net_params, 
        args.opt_type, 
        args.opt_param, 
        args.init_lr, 
        args.weight_decay, 
        args.no_decay_keys
    )
##################### VALIDATE FUNC ######################
    validate_func_dict = {
        'image_size_list'   : {224} if isinstance(args.image_size, int) else sorted({160, 224}),
        'ks_list'           : sorted({min(args.ks_list), max(args.ks_list)}),
        'expand_ratio_list' : sorted({min(args.expand_list), max(args.expand_list)}),
        'depth_list'        : sorted({min(net.depth_list), max(net.depth_list)}),
        'weight_quant_list' : ['fp32', 'SD4_per_channel'],
        'act_quant_list'    : ['fp32', 'int8']
    }
######################## PRETRAINED MODEL ########################
    if args.task == 'kernel':
        args.ofa_checkpoint_path = download_url(
            'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K7',
            model_dir='.torch/ofa_checkpoints/%d' % 0
        )
    elif args.task == 'depth':
        if args.phase == 1:
            args.ofa_checkpoint_path = download_url(
                'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K357',
                model_dir='.torch/ofa_checkpoints/%d' % 0
            )
        else:
            args.ofa_checkpoint_path = download_url(
                'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D34_E6_K357',
                model_dir='.torch/ofa_checkpoints/%d' % 0
            )
    elif args.task == 'expand':
        if args.phase == 1:
            args.ofa_checkpoint_path = download_url(
                'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D234_E6_K357',
                model_dir='.torch/ofa_checkpoints/%d' % 0
            )
        else:
            args.ofa_checkpoint_path = download_url(
                'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D234_E46_K357',
                model_dir='.torch/ofa_checkpoints/%d' % 0
            )
    elif args.task == 'bit':
        if args.phase == 1:
            args.ofa_checkpoint_path = download_url(
                "https://hanlab.mit.edu/files/OnceForAll/ofa_nets/ofa_mbv3_d234_e346_k357_w1.0",
                model_dir='.torch/ofa_checkpoints/%d' % 0
            )

    # Distributed dataparalllel
    net = nn.parallel.DistributedDataParallel(net, device_ids=[gpu], find_unused_parameters=True)
    if rank == 0:
        print('----- ENTER TRAINING PROCESS -----')
######################### TRAINING ######################
    if args.task == 'kernel':
        validate_func_dict["ks_list"] = sorted(args.ks_list)
        train(
            net, 
            gpu, 
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
    elif args.task == 'depth':
        train_elastic_depth(
            net,
            gpu,
            train_criterion,
            test_criterion,
            optimizer,
            args,
            run_config,  
            validate_func_dict
        )
    elif args.task == 'expand':
        train_elastic_expand(
            net,
            gpu,
            train_criterion,
            test_criterion, 
            optimizer,
            args, 
            run_config,
            validate_func_dict
        )
    elif args.task == 'bit':
        train_elastic_bit(
            net, 
            gpu, 
            train_criterion, 
            test_criterion, 
            optimizer, 
            args, 
            run_config, 
            validate_func_dict
        )
    else:
        raise NotImplementedError

def main(args):
    args = add_args(args)
    os.makedirs(args.path, exist_ok=True)

    os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]  = args.gpus
    os.environ['MASTER_ADDR']           = 'localhost'
    os.environ['MASTER_PORT']           = '8888'
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    args.world_size = len(args.gpus.split(','))
    mp.spawn(distributed_training, nprocs=len(args.gpus.split(',')), args=(args,))
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

    