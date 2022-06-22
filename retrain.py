import torch
from argparse import ArgumentParser, Namespace
from pathlib import Path
from utils import AverageMeter, DistributedMetric, cross_entropy_loss_with_soft_target
from utils import list_mean, subset_mean, val2list, accuracy, MyRandomResizedCrop
from utils import set_running_statistics
from int_quantization.fake_quantize import disable_observer, enable_observer

def retrain(model, subnet_setting):


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=Path, default=None, help="Path to model setting config")
    parser.add_argument('--pretrain', type=Path, default=None, help="Path to ofa pretrianed model")

def main():
    pass

if __name__ == '__main__':
    args = parse_args()
    main()