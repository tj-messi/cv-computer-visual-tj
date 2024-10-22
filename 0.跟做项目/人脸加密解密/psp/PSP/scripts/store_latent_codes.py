import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
import pickle

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset_with_pic_name import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp
from models.encoders import psp_encoders

import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(20)

def run():
    test_opts = TestOptions().parse()

    if test_opts.resize_factors is not None:
        assert len(
            test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
                                        'downsampling_{}'.format(test_opts.resize_factors))
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
                                        'downsampling_{}'.format(test_opts.resize_factors))
    else:
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    all_latent_codes={}
    for input_batch in tqdm(dataloader):
        with torch.no_grad():
            print(input_batch)
            input_cuda = input_batch[0].cuda().float()
            result_batch = run_on_batch(input_cuda, net, opts)
        for i in range(opts.test_batch_size):
            all_latent_codes[input_batch[1][i]]=result_batch[i]
    with open('./experiment/existing_faces.pkl', 'wb') as f:
        pickle.dump(all_latent_codes, f)

def run_on_batch(inputs, net, opts):
    encoder = net.encoder
    print("inputs",inputs)
    cur_latent_codes = encoder(inputs)
    cur_latent_codes += net.latent_avg.repeat(cur_latent_codes.shape[0], 1, 1)
    #print('latent_avg', net.latent_avg.repeat(cur_latent_codes.shape[0], 1, 1))
    print("cur_latent_codes", cur_latent_codes.shape, cur_latent_codes)
    return cur_latent_codes


if __name__ == '__main__':
    run()
