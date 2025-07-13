#!/usr/bin/env bash

OUTPUT_DIR='/media/tongji/VideoMAEv2-master/output/7-9-pt-alldata'
DATA_PATH='/media/tongji/VideoMAEv2-master/data/US_annotation(2class)/MAE2.csv'


# batch_size can be adjusted according to the graphics card
python -u run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.8 \
        --fname_tmpl 'img_{:05}.png' \
        --decoder_mask_type run_cell \
        --decoder_mask_ratio 0.5 \
        --decoder_depth 10 \
        --model pretrain_videomae_base_patch16_224 \
        --batch_size 4 \
        --with_checkpoint \
        --num_frames 16 \
        --sampling_rate 1 \
        --num_sample 2 \
        --num_workers 10 \
        --opt adamw \
        --lr 1e-3 \
        --min_lr 1e-12 \
        --clip_grad 0.02 \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 1 \
        --save_ckpt_freq 20 \
        --epochs 500 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
