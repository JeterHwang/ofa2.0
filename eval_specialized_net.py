# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
import os
import os.path as osp
import argparse
import math
import numpy as np
import random
from tqdm import tqdm

import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms, datasets

from utils import AverageMeter, DistributedMetric,accuracy
from model_zoo import ofa_specialized
from int_quantization import mappings, fake_quantize
from torch.quantization import QuantWrapper, add_quant_dequant, fuse_modules

specialized_network_list = [
    ################# FLOPs #################
    "flops@595M_top1@80.0_finetune@75",
    "flops@482M_top1@79.6_finetune@75",
    "flops@389M_top1@79.1_finetune@75",
    ################# ResNet50 Design Space #################
    "resnet50D_MAC@4.1B_top1@79.8",
    "resnet50D_MAC@3.7B_top1@79.7",
    "resnet50D_MAC@3.0B_top1@79.3",
    "resnet50D_MAC@2.4B_top1@79.0",
    "resnet50D_MAC@1.8B_top1@78.3",
    "resnet50D_MAC@1.2B_top1@77.1_finetune@25",
    "resnet50D_MAC@0.9B_top1@76.3_finetune@25",
    "resnet50D_MAC@0.6B_top1@75.0_finetune@25",
    ################# Google pixel1 #################
    "pixel1_lat@143ms_top1@80.1_finetune@75",
    "pixel1_lat@132ms_top1@79.8_finetune@75",
    "pixel1_lat@79ms_top1@78.7_finetune@75",
    "pixel1_lat@58ms_top1@76.9_finetune@75",
    "pixel1_lat@40ms_top1@74.9_finetune@25",
    "pixel1_lat@28ms_top1@73.3_finetune@25",
    "pixel1_lat@20ms_top1@71.4_finetune@25",
    ################# Google pixel2 #################
    "pixel2_lat@62ms_top1@75.8_finetune@25",
    "pixel2_lat@50ms_top1@74.7_finetune@25",
    "pixel2_lat@35ms_top1@73.4_finetune@25",
    "pixel2_lat@25ms_top1@71.5_finetune@25",
    ################# Samsung note10 #################
    "note10_lat@64ms_top1@80.2_finetune@75",
    "note10_lat@50ms_top1@79.7_finetune@75",
    "note10_lat@41ms_top1@79.3_finetune@75",
    "note10_lat@30ms_top1@78.4_finetune@75",
    "note10_lat@22ms_top1@76.6_finetune@25",
    "note10_lat@16ms_top1@75.5_finetune@25",
    "note10_lat@11ms_top1@73.6_finetune@25",
    "note10_lat@8ms_top1@71.4_finetune@25",
    ################# Samsung note8 #################
    "note8_lat@65ms_top1@76.1_finetune@25",
    "note8_lat@49ms_top1@74.9_finetune@25",
    "note8_lat@31ms_top1@72.8_finetune@25",
    "note8_lat@22ms_top1@70.4_finetune@25",
    ################# Samsung S7 Edge #################
    "s7edge_lat@88ms_top1@76.3_finetune@25",
    "s7edge_lat@58ms_top1@74.7_finetune@25",
    "s7edge_lat@41ms_top1@73.1_finetune@25",
    "s7edge_lat@29ms_top1@70.5_finetune@25",
    ################# LG G8 #################
    "LG-G8_lat@24ms_top1@76.4_finetune@25",
    "LG-G8_lat@16ms_top1@74.7_finetune@25",
    "LG-G8_lat@11ms_top1@73.0_finetune@25",
    "LG-G8_lat@8ms_top1@71.1_finetune@25",
    ################# 1080ti GPU (Batch Size 64) #################
    "1080ti_gpu64@27ms_top1@76.4_finetune@25",
    "1080ti_gpu64@22ms_top1@75.3_finetune@25",
    "1080ti_gpu64@15ms_top1@73.8_finetune@25",
    "1080ti_gpu64@12ms_top1@72.6_finetune@25",
    ################# V100 GPU (Batch Size 64) #################
    "v100_gpu64@11ms_top1@76.1_finetune@25",
    "v100_gpu64@9ms_top1@75.3_finetune@25",
    "v100_gpu64@6ms_top1@73.0_finetune@25",
    "v100_gpu64@5ms_top1@71.6_finetune@25",
    ################# Jetson TX2 GPU (Batch Size 16) #################
    "tx2_gpu16@96ms_top1@75.8_finetune@25",
    "tx2_gpu16@80ms_top1@75.4_finetune@25",
    "tx2_gpu16@47ms_top1@72.9_finetune@25",
    "tx2_gpu16@35ms_top1@70.3_finetune@25",
    ################# Intel Xeon CPU with MKL-DNN (Batch Size 1) #################
    "cpu_lat@17ms_top1@75.7_finetune@25",
    "cpu_lat@15ms_top1@74.6_finetune@25",
    "cpu_lat@11ms_top1@72.0_finetune@25",
    "cpu_lat@10ms_top1@71.1_finetune@25",
]
int_act_fake_quant = fake_quantize.default_fake_quant
log_weight_fake_quant_per_channel = fake_quantize.default_per_channel_log_weight_fake_quant
log_weight_fake_quant_per_tensor  = fake_quantize.default_log_weight_fake_quant

log4_per_channel_config = torch.quantization.QConfig(activation=int_act_fake_quant, weight=log_weight_fake_quant_per_channel)
log4_per_tensor_config  = torch.quantization.QConfig(activation=int_act_fake_quant, weight=log_weight_fake_quant_per_tensor)

def fuse_model(model : nn.Module):
    modules_to_fuse = []
    for m in model.named_children():
        if m[0] == 'blocks':
            modules_to_fuse.append(['blocks.0.conv.depth_conv.conv', 'blocks.0.conv.depth_conv.bn', 'blocks.0.conv.depth_conv.act'])
            modules_to_fuse.append(['blocks.0.conv.point_linear.conv', 'blocks.0.conv.point_linear.bn'])
            for i, residul_block in enumerate(m[1][1:]):
                if type(residul_block.conv.inverted_bottleneck.act) == torch.nn.ReLU:
                    modules_to_fuse.append([f"blocks.{i+1}.conv.inverted_bottleneck.conv", f"blocks.{i+1}.conv.inverted_bottleneck.bn", f"blocks.{i+1}.conv.inverted_bottleneck.act"])
                else:
                    modules_to_fuse.append([f"blocks.{i+1}.conv.inverted_bottleneck.conv", f"blocks.{i+1}.conv.inverted_bottleneck.bn"])

                if type(residul_block.conv.depth_conv.act) == torch.nn.ReLU:
                    modules_to_fuse.append([f"blocks.{i+1}.conv.depth_conv.conv", f"blocks.{i+1}.conv.depth_conv.bn", f"blocks.{i+1}.conv.depth_conv.act"])
                else:
                    modules_to_fuse.append([f"blocks.{i+1}.conv.depth_conv.conv", f"blocks.{i+1}.conv.depth_conv.bn"])
            
                modules_to_fuse.append([f"blocks.{i+1}.conv.point_linear.conv", f"blocks.{i+1}.conv.point_linear.bn"])

    return fuse_modules(model, modules_to_fuse)

def add_Qconfig(model : nn.Module, qconfig):
    for m in model.named_children():
        if m[0] == 'blocks':
            #m[1][0].conv.depth_conv.conv.qconfig = qconfig
            m[1][0].conv.point_linear.conv.qconfig = qconfig
            for residual_block in m[1][1:]:
                residual_block.conv.inverted_bottleneck.conv.qconfig = qconfig
                # residual_block.conv.depth_conv.conv.qconfig = qconfig
                residual_block.conv.point_linear.conv.qconfig = qconfig 
def add_quant_dequant(model : nn.Module):
    for name, child in model.named_children():
        if name == 'blocks':
            for idx, block in enumerate(child):
                if idx == 0:
                    #block.conv.depth_conv._modules['conv'] = QuantWrapper(block.conv.depth_conv.conv)
                    block.conv.point_linear._modules['conv'] = QuantWrapper(block.conv.point_linear.conv)
                else:
                    block.conv.inverted_bottleneck._modules['conv'] = QuantWrapper(block.conv.inverted_bottleneck.conv)
                    #block.conv.depth_conv._modules['conv'] = QuantWrapper(block.conv.depth_conv.conv)
                    block.conv.point_linear._modules['conv'] = QuantWrapper(block.conv.point_linear.conv)
def print_size_of_model(model):
    """ Print the size of the model.
    
    Args:
        model: model whose size needs to be determined

    """
    model = model.cpu()
    torch.save(model.state_dict(), "temp.p")
    print('Size of the model(MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def same_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.enabled = True
    #torch.backends.cudnn.deterministic = True

def train_one_epoch(model, train_loader, optimizer, criterion, lr_scheduler, device, epoch, args):
    distributed = isinstance(model, nn.parallel.DistributedDataParallel)
    if distributed:
        losses = DistributedMetric("train_loss", args.world_size)
        top1 = DistributedMetric('top1', args.world_size)
        top5 = DistributedMetric('top5', args.world_size)
    else:
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
    
    model.train()
    with tqdm(total=len(train_loader), desc='Train Epoch #{}'.format(epoch + 1), disable=distributed and (device != 0)) as t:
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            if (step + 1) % args.gradient_accumulation_steps != 0:
                with model.no_sync():
                    output = model(images)
                    loss = criterion(output, labels) / args.gradient_accumulation_steps
                    loss.backward()
            else:
                output = model(images)
                loss = criterion(output, labels) / args.gradient_accumulation_steps
                loss.backward()
            
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(torch.tensor(loss.item(), device=loss.device), torch.tensor(images.size(0), device=loss.device))
            top1.update(torch.tensor(acc1[0].item(), device=acc1[0].device), torch.tensor(images.size(0), device=acc1[0].device))
            top5.update(torch.tensor(acc5[0].item(), device=acc5[0].device), torch.tensor(images.size(0),  device=acc5[0].device))
                
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                ##### Gradient Clipping #####
                if args.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            t.set_postfix(
                {
                    "loss": losses.avg.item(),
                    "top1": top1.avg.item(),
                    "top5": top5.avg.item(),
                    "lr" : optimizer.param_groups[0]["lr"],
                    "img_size": images.size(2),
                }
            )
            t.update(1)

def test(model, data_loader, criterion, device, task, image_size, epoch, args, batch_num=None):
    distributed = isinstance(model, nn.parallel.DistributedDataParallel)
    if distributed:
        losses = DistributedMetric("train_loss", args.world_size)
        top1 = DistributedMetric('top1', args.world_size)
        top5 = DistributedMetric('top5', args.world_size)
    else:
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        if epoch >= args.num_observer_update_epochs:
            # print("Disabling observer for subseq epochs, epoch = ", epoch)
            model.apply(fake_quantize.disable_observer)
        if epoch >= args.num_batch_norm_update_epochs:
            # print("Freezing BN for subseq epochs, epoch = ", epoch)
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        with tqdm(total=min(batch_num, len(data_loader)) if batch_num is not None else len(data_loader), desc=task, disable=distributed and (device != 0)) as t:
            for i, (images, labels) in enumerate(data_loader):
                if batch_num is not None and i == batch_num:
                    break
                images, labels = images.cuda(), labels.cuda()    
                # compute output
                output = model(images)
                loss = criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                losses.update(torch.tensor(loss.item(), device=loss.device), torch.tensor(images.size(0), device=loss.device))
                top1.update(torch.tensor(acc1[0].item(), device=acc1[0].device), torch.tensor(images.size(0), device=acc1[0].device))
                top5.update(torch.tensor(acc5[0].item(), device=acc5[0].device), torch.tensor(images.size(0),  device=acc5[0].device))
                
                t.set_postfix(
                    {
                        "loss": losses.avg.item(),
                        "top1": top1.avg.item(),
                        "top5": top5.avg.item(),
                        "img_size": images.size(2),
                    }
                )
                t.update(1)
    if distributed and device == 0:
        print(f"({task}) Test OFA specialized net <{args.net}> with image size {image_size}:")
        print(f"({task}) Results: loss={round(losses.avg.item(), 5)},\t top1={round(top1.avg.item(), 1)},\t top5={round(top5.avg.item(), 1)}")


def PTQ(args):
    device_list = [int(_) for _ in args.gpus.split(",")]
    torch.cuda.set_device(device_list[0])

    same_seed(args.seed)
    net, image_size = ofa_specialized(net_id=args.net, pretrained=True)
    
    print_size_of_model(net)

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            osp.join(args.path, "train"),
            transforms.Compose(
                [
                    transforms.Resize(int(math.ceil(image_size / 0.875))),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            osp.join(args.path, "val"),
            transforms.Compose(
                [
                    transforms.Resize(int(math.ceil(image_size / 0.875))),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    criterion = nn.CrossEntropyLoss()

    net.eval()
    net = fuse_model(net)
    test(net, test_loader, criterion, 'gpu', 'Test FP32', image_size, args)

    add_quant_dequant(net)
    add_Qconfig(net, log4_per_channel_config)
    torch.quantization.prepare(net, inplace=True)
    test(net, train_loader, criterion, 'gpu', 'Calibrate', image_size, args, args.batch_num)

    net.cpu()
    net.eval()
    torch.quantization.convert(net.blocks, inplace=True)
    # net.cuda()
    # print(net.classifier.linear.weight)
    # print(net.blocks[19].conv.inverted_bottleneck.conv.quant)
    # print(net.blocks[19].conv.inverted_bottleneck.conv.module.weight)
    test(net, test_loader, criterion, 'cpu', 'Test Quantized', image_size, args)
    print_size_of_model(net)

def QAT(gpu, args):
    rank = gpu
    dist.init_process_group(
    	backend     = args.backend,
        init_method = 'env://',
    	world_size  = args.world_size,
    	rank        = rank
    )
    torch.cuda.set_device(gpu)
    same_seed(args.seed)
    
    
    net, image_size = ofa_specialized(net_id=args.net, pretrained=True)
    if rank == 0:
        print_size_of_model(net)
    
    train_dataset = datasets.ImageFolder(
        osp.join(args.path, "train"),
        transforms.Compose(
            [
                transforms.Resize(int(math.ceil(image_size))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    test_dataset = datasets.ImageFolder(
        osp.join(args.path, "val"),
        transforms.Compose(
            [
                transforms.Resize(int(math.ceil(image_size / 0.875))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
    	test_dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=args.eval_batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    
    criterion = nn.CrossEntropyLoss()
    # Fuse model
    net.eval()
    net = fuse_model(net)

    # Add Qconfig
    add_quant_dequant(net)
    add_Qconfig(net, log4_per_channel_config)
    
    torch.quantization.prepare_qat(net, inplace=True)
    #print(net)
    if args.sync_bn:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    optimizer = torch.optim.SGD(
        net.parameters(), 
        lr=1,
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    # Cosin Anealing scheduler with warm-up
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps if len(train_loader) % args.gradient_accumulation_steps == 0 else len(train_loader) // args.gradient_accumulation_steps + 1
    warm_up_step = args.warmup_epochs * steps_per_epoch
    total_step = args.n_epochs * steps_per_epoch
    lr_min, lr_max = 1e-5, args.base_lr
    lambda0 = lambda cur_step: cur_step / warm_up_step * (lr_max - lr_min) + lr_min  if cur_step < warm_up_step else \
        (lr_min + 0.5 * (lr_max- lr_min) * (1 + math.cos((cur_step - warm_up_step) / (total_step - warm_up_step) * math.pi)))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    
    # Wrapp the model 
    net.cuda()
    net = nn.parallel.DistributedDataParallel(net, device_ids=[gpu])
    
    for epoch in range(args.n_epochs):
        # if rank == 0:
        #     print(f'---- trainning on epoch {epoch} ----')
        train_sampler.set_epoch(epoch)
        train_one_epoch(net, train_loader, optimizer, criterion, lr_scheduler, gpu, epoch, args)

        #net.apply(fake_quantize.disable_observer)
        test(net, test_loader, criterion, gpu, 'QAT', image_size, epoch, args)
        #net.apply(fake_quantize.enable_observer)
            

    # Convert model and inference on CPU

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", help="The path of imagenet", type=str, default="D:\\IC-design\\Project\\Tzi-Dar Chiueh\\AutoML\\ofa\\dataset\\imagenet"
    )
    parser.add_argument("-g", "--gpus", help="The gpu(s) to use", type=str, default="0")
    parser.add_argument(
        "-b",
        "--batch_size",
        help="The batch on every device for validation",
        type=int,
        default=32,
    )
    parser.add_argument("-j", "--workers", help="Number of workers", type=int, default=6)
    parser.add_argument(
        "-n",
        "--net",
        metavar="NET",
        default="flops@595M_top1@80.0_finetune@75",
        choices=specialized_network_list,
        help="OFA specialized networks: "
        + " | ".join(specialized_network_list)
        + " (default: pixel1_lat@143ms_t op1@80.1_finetune@75)",
    )
    parser.add_argument("-s", "--seed", help="The seed", type=int, default=42)
    parser.add_argument("--batch_num", help="Calibration batch number", type=int, default=10)
    parser.add_argument("--mode", help="QAT or PTQ", type=str, default="QAT", choices=['QAT', 'PTQ'])
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--n_epochs', type=int, default=93)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument("--sync_bn", type=bool, help="Use sync batch norm", default=False)
    parser.add_argument("--backend", type=str, help="DDP backend", default='gloo')
    parser.add_argument("--num_observer_update_epochs", type=int, default=4)
    parser.add_argument("--num_batch_norm_update_epochs", type=int, default=3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    device_list = [int(_) for _ in args.gpus.split(",")]
    if args.mode == 'QAT':
        args.real_batch_size = args.batch_size * max(len(device_list), 1) * args.gradient_accumulation_steps
        args.world_size = len(device_list)

        os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]  = args.gpus
        os.environ['MASTER_ADDR']           = 'localhost'
        os.environ['MASTER_PORT']           = '8888'

        mp.spawn(QAT, nprocs=len(device_list), args=(args,))
    else:
        PTQ(args)




