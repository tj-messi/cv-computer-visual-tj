#!/usr/bin/env bash
set -x

export MASTER_PORT=34001
export OMP_NUM_THREADS=1

OUTPUT_DIR='/media/tongji/VideoMAEv2-master/output/test_output'
DATA_PATH='/media/tongji/VideoMAEv2-master/data/US_annotation'
MODEL_PATH='/media/tongji/VideoMAEv2-master/output/US_origin_ft/checkpoint-best/mp_rank_00_model_states.pt'


# 8 for 1 node, 16 for 2 node, etc.

# batch_size can be adjusted according to the graphics card
python test_evaluation.py \
        --model vit_base_patch16_224 \
        --validation \
        --data_set SSV2 \
        --nb_classes 2 \
        --fname_tmpl 'img_{:05}.png' \
        --start_idx 1 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 4 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 10 \
        --opt adamw \
        --lr 6e-4 \
        --drop_path 0.25 \
        --layer_decay 0.9 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --warmup_epochs 15 \
        --epochs 500 \
        --test_num_segment 2 \
        --test_num_crop 3 \
        --dist_eval --enable_deepspeed \
