#!/bin/bash

GPUS_PER_NODE=`nvidia-smi -L | wc -l`
OUTPUT_DIR=OUTPUT/videomae2_vit_small_with_vit_base_teacher_k400_epoch_400/train${NUM_TRAIN}_val_${NUM_VAL}/mask${MASK_RATIO}_BS${BATCH_SIZE}
rm -rf $OUTPUT_DIR/*
sleep 1
DATA_PATH=${SRC_DIR}/train_list_video_trn${NUM_TRAIN}_val${NUM_VAL}_items.csv
DATA_ROOT=${DST_DIR}

echo $(hostname) $MASTER_ADDR $SLURM_JOB_NODELIST ${MASTER_PORT}

# pretrain
torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
    run_mae_pretraining.py \
    --data_root ${DATA_ROOT} \
    --data_path ${DATA_PATH} \
    --mask_type tube \
    --mask_ratio ${MASK_RATIO} \
    --decoder_mask_type run_cell \
    --decoder_mask_ratio 0.5 \
    --model pretrain_videomae_base_patch16_224 \
    --decoder_depth 4 \
    --batch_size ${BATCH_SIZE} \
    --num_sample 4 \
    --num_frames 16 \
    --sampling_rate 4 \
    --num_workers 8 \
    --lr 1e-3 \
    --opt adamw \
    --opt_betas 0.9 0.95 \
    --warmup_epochs 20 \
    --save_ckpt_freq 20 \
    --epochs 200 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --save_ckpt_freq 1





















