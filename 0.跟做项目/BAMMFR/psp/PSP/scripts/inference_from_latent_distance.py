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

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    f = open(stats_path, 'w+')
    for feature_dim in range(1, 19):
        os.makedirs(os.path.join(out_path_results, 'feature_dim_' + str(feature_dim)), exist_ok=True)
        os.makedirs(os.path.join(out_path_coupled, 'feature_dim_' + str(feature_dim)), exist_ok=True)
        global_i = 0
        global_time = []
        match_correct_num = 0
        for input_batch in tqdm(dataloader):
            if global_i >= opts.n_images:
                break
            with torch.no_grad():
                input_cuda = input_batch[0].cuda().float()
                tic = time.time()
                result_batch = run_on_batch(input_cuda, net, feature_dim)
                toc = time.time()
                global_time.append(toc - tic)

            for i in range(opts.test_batch_size):
                result = tensor2im(result_batch[0][i])
                im_path = dataset.paths[global_i]
                print('gt:', im_path)
                print('pred:', result_batch[1])
                print()
                # if os.path.basename(im_path).split('_')[1] == os.path.basename(result_batch[1]).split('_')[1]:
                if os.path.basename(im_path) == os.path.basename(result_batch[1]):
                    match_correct_num += 1

                if opts.couple_outputs or global_i % 100 == 0:
                    input_im = log_input_image(input_batch[0][i], opts)
                    resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
                    if opts.resize_factors is not None:
                        # for super resolution, save the original, down-sampled, and output
                        source = Image.open(im_path)
                        source = source.convert("RGB")
                        res = np.concatenate([np.array(source.resize(resize_amount)),
                                              np.array(input_im.resize(resize_amount, resample=Image.NEAREST)),
                                              np.array(result.resize(resize_amount))], axis=1)
                    else:
                        # otherwise, save the original and output
                        res = np.concatenate([np.array(input_im.resize(resize_amount)),
                                              np.array(result.resize(resize_amount))], axis=1)
                    Image.fromarray(res).save(
                        os.path.join(out_path_coupled, 'feature_dim_' + str(feature_dim), os.path.basename(im_path)))

                im_save_path = os.path.join(out_path_results, 'feature_dim_' + str(feature_dim),
                                            os.path.basename(im_path))
                Image.fromarray(np.array(result)).save(im_save_path)

                global_i += 1

        result_str = 'feature_dim_used: {:d}  total: {:d}  match: {:d}  acc: {:.4f}\n'.format(feature_dim, global_i,
                                                                                              match_correct_num,
                                                                                              match_correct_num / global_i)
        print(result_str)
        f.write(result_str)


def run_on_batch(inputs, net, feature_dim):
    encoder = net.encoder
    cur_latent_codes = encoder(inputs)
    cur_latent_codes += net.latent_avg.repeat(cur_latent_codes.shape[0], 1, 1)

    with open('./experiment/existing_faces.pkl', 'rb') as f:
        history_latent_codes = pickle.load(f)

    lowest_distance = 999
    shredhold = 9999
    wanted_pic_name = ''
    wanted_latent_codes = []

    for pic_name in history_latent_codes:
        pdist = torch.nn.PairwiseDistance(p=2, eps=1e-06)

        #distance = pdist(torch.flatten(cur_latent_codes[0][:feature_dim]).unsqueeze(0),torch.flatten(history_latent_codes[pic_name][:feature_dim]).unsqueeze(0))
        distance = pdist(torch.flatten(cur_latent_codes[0][18-feature_dim:]).unsqueeze(0),torch.flatten(history_latent_codes[pic_name][18-feature_dim:]).unsqueeze(0))
        #print(distance)
        distance = distance[0]
        if distance < lowest_distance:
            lowest_distance = distance
            wanted_pic_name = pic_name
            wanted_latent_codes = history_latent_codes[pic_name]
    wanted_latent_codes = wanted_latent_codes.unsqueeze(0)
    if False or lowest_distance > shredhold or lowest_distance == 999:
        wanted_pic_name = ''
        wanted_latent_codes = cur_latent_codes
    print(lowest_distance)
    images, result_latent = net.decoder([wanted_latent_codes], randomize_noise=False, input_is_latent=True)
    images = net.face_pool(images)
    return images, wanted_pic_name


if __name__ == '__main__':
    run()

