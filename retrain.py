import torch
import os
import random
import torch.multiprocessing as mp
import numpy as np
import torch.distributed as dist
from pathlib import Path
from argparse import ArgumentParser

from networks import OFAMobileNetV3, MobileNetV3Large
from training import load_pretrained_model, get_cost, save_model, validate, train_one_epoch
from run_manager.run_config import DistributedImageNetRunConfig
from training.progressive_shrinking import reset_running_statistics
from utils import build_optimizer, cross_entropy_with_label_smoothing
from utils import MyRandomResizedCrop
from utils import download_url, init_lsq

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

def _build_optimizer(net, args):
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
    return optimizer

def retrain(net, run_config, train_criterion, test_criterion, args):
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
    optimizer = _build_optimizer(net, args)
    best_acc = 0
    for epoch in range(args.n_epochs + args.warmup_epochs):
        train_loss, (train_top1, train_top5) = train_one_epoch(
			net, 
			None,
			train_criterion,
			optimizer,
			run_config,
			args, 
			epoch, 
			args.warmup_epochs, 
			args.warmup_lr
		)
        if (epoch + 1) % args.validation_frequency == 0:
            LOSS, (TOP1, TOP5) = validate(
		        net, 
		        test_criterion,
		        run_config,
		        args,
                epoch = epoch,
		        is_test = True,
                run_str = "Validate Random Subnet",
	        )
            is_best = TOP1 > best_acc
            best_acc = max(best_acc, TOP1)
            if torch.distributed.get_rank() == 0:
                save_model(
					args.save_path,
					net,
					optimizer,
					run_config,
					epoch,
					best_acc,
					is_best=is_best
				)
            torch.distributed.barrier()

def subnet_sampling(gpu, args):
    rank = gpu
    torch.cuda.set_device(gpu)
    dist.init_process_group(
    	backend     = args.backend,
        init_method = 'env://',
    	world_size  = args.world_size,
    	rank        = rank
    )
    
    same_seed(args.manual_seed)
    MyRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size
    run_config = DistributedImageNetRunConfig(
        n_epochs                = args.n_epochs, 
        init_lr                 = args.init_lr, 
        lr_schedule_type        = args.lr_schedule_type, 
        train_batch_size        = args.base_batch_size,
		test_batch_size         = args.base_batch_size * 4, 
        train_size              = args.train_size,
        valid_size              = args.valid_size, 
        n_worker                = args.n_worker,
		resize_scale            = args.resize_scale, 
        distort_color           = args.distort_color, 
        image_size              = args.image_size, 
        num_replicas            = args.world_size, 
        save_path               = args.dataset_path,  
        rank                    = rank
    )
    dynamic_net = OFAMobileNetV3(
        n_classes           = run_config.data_provider.n_classes, 
        bn_param            = (args.bn_momentum, args.bn_eps),
        dropout_rate        = args.dropout, 
        base_stage_width    = args.base_stage_width, 
        width_mult          = args.width_mult_list,
        ks_list             = args.ks_list, 
        expand_ratio_list   = args.expand_list, 
        depth_list          = args.depth_list,
        weight_quant_list   = ['fp32', 'lsq4_per_channel', 'lsq3_per_channel'],
		act_quant_list		= ['fp32', 'lsq4_per_tensor', 'lsq3_per_tensor'],
    ).cuda()
    dynamic_net.apply(init_lsq)
    load_pretrained_model(dynamic_net, args.ofa_ckpt_path)
    dynamic_net = torch.nn.parallel.DistributedDataParallel(
        dynamic_net, 
        device_ids=[gpu], 
        find_unused_parameters=True
    )
    train_criterion = lambda pred, target: cross_entropy_with_label_smoothing(pred, target, args.label_smoothing)
    test_criterion = torch.nn.CrossEntropyLoss()
    
    random_subnet_setting = dynamic_net.module.sample_active_subnet()
    run_config.data_provider.assign_active_img_size(224)
    reset_running_statistics(
        net=dynamic_net,
        run_config=run_config,
        distributed=False
    )
    loss, (top1, top5) = validate(
		dynamic_net, 
		test_criterion,
		run_config,
		args,
		is_test = True,
        run_str = "Validate Random Dynamic Subnet",
	)
    bitOps_w, params_w = get_cost(dynamic_net, 224, random_subnet_setting)
    torch.distributed.barrier()

    net = dynamic_net.module.get_active_subnet()
    net = torch.nn.parallel.DistributedDataParallel(
        net, 
        device_ids=[gpu], 
        find_unused_parameters=True
    )
    LOSS, (TOP1, TOP5) = validate(
		dynamic_net, 
		test_criterion,
		run_config,
		args,
		is_test = True,
        run_str = "Validate Random Subnet",
	)
    if torch.distributed.get_rank() == 0:
        print(f"===== Random Subnet accuracy({top1}) =====")
        print(f"Bitops = {bitOps_w} (G)")
        print(f"Param size = {params_w} (MB)")
        print(f"Dynamic Net Acc = {top1 * 100}%")
        print(f"Subnet Acc = {TOP1 * 100}%")
        print(f"===================================================")

def parse_args():
    parser = ArgumentParser()
    ''' System Related '''
    parser.add_argument('--ofa_ckpt_path', type=Path, default='./ckpt/Epoch-7.pt', help="Path to ofa pretrianed model")
    parser.add_argument('--save_path', type=Path, default='./subnets', help="Path to save sampled subnets")
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument("--gpus", type=str, help="gpu devices id", default="0,1,2")
    parser.add_argument('--manual_seed', type=int, default=0)

    ''' Data Related '''
    parser.add_argument('--dataset_path', type=Path, default='/home/ntu329/Documents/Datasets/ImageNet')
    parser.add_argument('--continuous_size', type=bool, default=True)
    parser.add_argument('--not_sync_distributed_image_size', type=bool, default=False)
    
    ''' Network Related '''
    parser.add_argument('--bn_momentum', type=float, default=0.1)
    parser.add_argument('--bn_eps', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--base_stage_width', type=str, default='proxyless')
    parser.add_argument('--width_mult_list', type=list, default=[1.0])
    parser.add_argument('--ks_list', type=list, default=[3,5,7])
    parser.add_argument('--expand_list', type=list, default=[3,4,6])
    parser.add_argument('--depth_list', type=list, default=[2,3,4])
    parser.add_argument('--weight_quant_list', type=list, default=['fp32', 'lsq4_per_channel', 'lsq3_per_channel'])
    parser.add_argument('--act_quant_list', type=list, default=['fp32', 'lsq4_per_tensor', 'lsq3_per_tensor'])    

    ''' Testing Related '''
    parser.add_argument('--base_batch_size', type=int, default=64)
    parser.add_argument('--train_size', type=float, default=0.5)
    parser.add_argument('--valid_size', type=float, default=0.05)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--resize_scale', type=float, default=0.08)
    parser.add_argument('--distort_color', type=str, default='tf')
    parser.add_argument('--image_size', type=list, default=[128,160,192,224])
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    ''' Retrain Related '''
    parser.add_argument('--kd_ratio', type=float, default=1.0)
    parser.add_argument('--kd_type', type=str, default='ce')
    parser.add_argument('--n_epochs', type=int, default=8)
    parser.add_argument('--init_lr', type=float, default=0.005)
    parser.add_argument('--lr_schedule_type', type=str, default='cosine')
    parser.add_argument('--validation_frequency', type=int, default=1)
    
    args = parser.parse_args()
    return args

def main(args):
    args.save_path.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_DEVICE_ORDER"]       = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]    = args.gpus
    os.environ['MASTER_ADDR']             = 'localhost'
    os.environ['MASTER_PORT']             = '8888'
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    
    args.world_size = len(args.gpus.split(','))
    args.width_mult_list    = (
        args.width_mult_list[0]
        if len(args.width_mult_list) == 1
        else args.width_mult_list
    ) 
    args.teacher_path = download_url(
        'https://hanlab.mit.edu/files/OnceForAll/ofa_checkpoints/ofa_D4_E6_K7',
        model_dir='.torch/ofa_checkpoints/0'
    )
    mp.spawn(subnet_sampling, nprocs=len(args.gpus.split(',')), args=(args,))

if __name__ == '__main__':
    args = parse_args()
    main(args)