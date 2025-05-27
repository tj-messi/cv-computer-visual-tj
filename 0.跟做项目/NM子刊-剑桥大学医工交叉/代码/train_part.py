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
import os
from guided_diffusion.train_util import TrainLoop
import numpy as np


def train_fun(args,log_dir,gene_order_path,n):
    for i in range(args.all_gene//args.gene_num):
        # Calculate the gene group range for this iteration
        gene_start = (n-1)*args.all_gene+(i*args.gene_num)
        gene_end = (n-1)*args.all_gene+((i+1)*args.gene_num)
        
        # Create the base directory path (without timestamp)
        base_dir = log_dir + args.dataset_use + '/' + str(args.SR_times) + 'X' + '/G' + str(gene_start) + '-' + str(gene_end)
        
        # Check if this gene group has already been trained
        already_trained = False
        
        # Look for existing directories matching this gene group pattern
        if os.path.exists(log_dir + args.dataset_use + '/' + str(args.SR_times) + 'X/'):
            print(log_dir + args.dataset_use + '/' + str(args.SR_times) + 'X/')
            existing_dirs = [d for d in os.listdir(log_dir + args.dataset_use + '/' + str(args.SR_times) + 'X/')
                            if d.startswith('G' + str(gene_start) + '-' + str(gene_end))]
            
            # Check each matching director
            # y for PT files
            for dir_name in existing_dirs:
                full_dir_path = os.path.join(log_dir + args.dataset_use + '/' + str(args.SR_times) + 'X/', dir_name)

                if os.path.isdir(full_dir_path):
                    # Get all PT files in this directory
                    pt_files = [f for f in os.listdir(full_dir_path) if f.endswith('.pt')]
                    
                                    # Check if any PT file has non-zero last three digits
                    import re
                    for pt_file in pt_files:

                        # Extract the model number from filename (assuming format like "model019802.pt")
                        match = re.search(r'model(\d+)\.pt', pt_file)
                        if match:
                            model_num = match.group(1)
                            # Check if the last three digits are not all zeros
                            if model_num[-3:] != '000':
                                already_trained = True
                                logger.log(f"Skipping gene group G{gene_start}-{gene_end} as it has a completed model file ({pt_file}) in {full_dir_path}")
                                break
                    
        if already_trained:
            continue
        
        # Otherwise, proceed with training
        loop_start_time = time.time()
        gene_order = np.load(gene_order_path)[gene_start:gene_end]
        
        # Create new directory with timestamp
        cur_time = time.strftime('%m%d-%H%M', time.localtime())
        save_dir = base_dir + '_{}'.format(cur_time)
        logger.configure(dir=save_dir+'/')

        logger.log(f"Training gene group G{gene_start}-{gene_end}...")
        logger.log("creating data loader...")
        # Load the super-resolution dataset
        brain_dataset = load_superres_data(args.data_root, args.dataset_use, status='Train', SR_times=args.SR_times,
                                        gene_num=args.gene_num, all_gene=args.all_gene, gene_order=gene_order)
        
        logger.log("creating model...")
        model, diffusion = sr_create_model_and_diffusion(args)

        model.to(dist_util.dev())
        # Create the schedule sampler based on the chosen method
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        logger.log("training...")
        # Start the training loop
        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=brain_dataset,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            SR_times=args.SR_times,
            epoch = args.epoch
        ).run_loop()
        
        # Calculate and log the time taken for this loop
        loop_duration = time.time() - loop_start_time
        hours = int(loop_duration // 3600)
        minutes = int((loop_duration % 3600) // 60)
        logger.log(f"循环 {i} 完成，耗时: {hours}小时{minutes}分钟")

def load_superres_data(data_root,dataset_use,status,SR_times,gene_num,all_gene,gene_order):
    # Load the super-resolution data using the specified directories
    return load_data(data_root=data_root,dataset_use=dataset_use,status=status,SR_times=SR_times,gene_num=gene_num,all_gene=all_gene,gene_order=gene_order)
