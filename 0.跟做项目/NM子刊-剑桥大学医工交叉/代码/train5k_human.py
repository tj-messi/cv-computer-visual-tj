import argparse
import yaml
import argparse, time, random
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_create_model_and_diffusion,
    add_dict_to_argparser
)
import torch
import os
from guided_diffusion.train_util import TrainLoop
import numpy as np
from mpi4py import MPI
from train_part import train_fun
comm =MPI.COMM_WORLD
rank = comm.Get_rank()

# 定义 GPU ID 列表，根据 rank 来选择对应的 GPU
gpu_ids = [6]  # GPU 0 和 GPU 1
torch.cuda.set_device(gpu_ids[rank])

def main():
    # Parse command-line arguments and set up distributed training
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    args.all_gene = 1000 #change
    args.gene_num = 20 #change
    args.batch_size= 8 #change
    args.SR_times= 5
    args.dataset_use = 'Xenium5k_human'
    args.epoch = 800
    gene_order_path = os.path.join(args.data_root, args.dataset_use+'/gene_order1.npy')
    n=1
    #n=xy1,2[01] zc 34[23]  xx 56[45] 
    log_dir = 'logs5K/'
    train_fun(args,log_dir,gene_order_path,n)

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--./config/config_train.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()

    # Load the configuration from the YAML file
    with open('./config/config_train.yaml', "r") as file:
        config = yaml.safe_load(file)

    # Add the configuration values to the argument parser
    add_dict_to_argparser(parser, config)

    return parser


if __name__ == "__main__":

    main()

