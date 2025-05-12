import argparse
import yaml
import argparse, time, random
from guided_diffusion import dist_util, logger
from guided_diffusion.img import load_data
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
comm =MPI.COMM_WORLD
rank = comm.Get_rank()

# 定义 GPU ID 列表，根据 rank 来选择对应的 GPU
gpu_ids = [1]  # GPU 0 和 GPU 1
torch.cuda.set_device(gpu_ids[rank])
from train_part1 import train_fun
def main():
    # Parse command-line arguments and set up distributed training
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    args.all_gene = 100 #change
    args.gene_num = 20 #change
    args.batch_size= 4 #change
    args.SR_times= 10
    args.dataset_use = 'VisiumHD_mouseembryo_sorted_data1'
    args.epoch = 500
    args.data_root = '/media/cbtil/T7 Shield/NMI/data/'
    gene_order_path = os.path.join(args.data_root, args.dataset_use+'/gene_order.npy')
    genename_path=os.path.join(args.data_root, args.dataset_use+'/gene_names.txt')
    n=56
    log_dir = 'logsVisiumhdmouseembryo/'
    #n=xy1,2[01] zc 34[23]  xx 56[45] 
    train_fun(args,log_dir,gene_order_path,genename_path,n)

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--./config/config_train.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()

    # Load the configuration from the YAML file
    with open('/media/cbtil/T7 Shield/NMI/code/config/config_train.yaml', "r") as file:
        config = yaml.safe_load(file)

    # Add the configuration values to the argument parser
    add_dict_to_argparser(parser, config)

    return parser


if __name__ == "__main__":

    main()

