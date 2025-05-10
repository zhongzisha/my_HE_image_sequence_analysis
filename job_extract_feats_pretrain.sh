#!/bin/bash

#SBATCH --job-name=debug2
#SBATCh --mail-type=FAIL
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100x:1,lscratch:100
#SBATCH --time=200:00:00
##SBATCH --exclusive
#SBATCH --output=%x-%j.out
#SBATCH --export=ALL


if [ "$CLUSTER_NAME" == "FRCE" ]; then
    source $FRCE_DATA_ROOT/anaconda3/bin/activate th26
    module load cuda/11.8
    module load cudnn/8.8.3-cuda11
    module load gcc/11.2.0

    cd $FRCE_DATA_ROOT/VideoMAEv2/
    export MYTMP_DIR=/tmp/zhongz2/
    export SRC_DIR=/mnt/gridftp/zhongz2/tcga_ffpe_all/patch_videos
else
    source /data/zhongz2/anaconda3/bin/activate th24
    module load CUDA/12.1
    module load cuDNN/8.9.2/CUDA-12
    module load gcc/11.3.0

    cd /home/zhongz2/VideoMAEv2
    export MYTMP_DIR=/lscratch/$SLURM_JOB_ID
    export SRC_DIR=/data/zhongz2/tcga_ffpe_all/patch_videos
fi

NUM_TRAIN=${1}
NUM_VAL=${2}
MASK_RATIO=${3}
BATCH_SIZE=${4}
MODEL_PATH=`ls OUTPUT/videomae2_vit_small_with_vit_base_teacher_k400_epoch_400/train${NUM_TRAIN}_val_${NUM_VAL}/mask${MASK_RATIO}_BS${BATCH_SIZE}/checkpoint*.pth -alth | head -n 1 | awk '{ print $9 }'`

srun python extract_tad_feature_pretrain.py \
--ckpt_path ${MODEL_PATH}

if [ $SLURM_PROCID -eq 0 ]; then
subset="val"
IMG_KEY="video"
export DST_DIR=${MYTMP_DIR}/videomaev2_${NUM_TRAIN}_${NUM_VAL}
mkdir -p ${DST_DIR}
bash extract_tars.sh ${SRC_DIR}/${subset}_list_${IMG_KEY}_trn${NUM_TRAIN}_val${NUM_VAL}.txt ${SRC_DIR} ${DST_DIR}/${subset}

python extract_tad_feature_pretrain.py \
--ckpt_path ${MODEL_PATH} \
--action step2


# path to pretrain model
MODEL_PATH='/scratch/cluster_scratch/zhongz2/VideoMAEv2/OUTPUT/pretrain/checkpoint-199.pth'
MODEL_PATH=OUTPUT/videomae2_vit_small_with_vit_base_teacher_k400_epoch_400/train${NUM_TRAIN}_val_${NUM_VAL}/mask${MASK_RATIO}_BS${BATCH_SIZE}/checkpoint-15.pth
MODEL_PATH=`ls OUTPUT/videomae2_vit_small_with_vit_base_teacher_k400_epoch_400/train${NUM_TRAIN}_val_${NUM_VAL}/mask${MASK_RATIO}_BS${BATCH_SIZE}/checkpoint*.pth -alth | head -n 1 | awk '{ print $9 }'`
# Set the path to save video
SAVE_PATH="${MODEL_PATH:0:$((${#MODEL_PATH} - 4))}_results"
mkdir -p $SAVE_PATH;

for d in `ls ${DST_DIR}/val/`;do
d=${DST_DIR}/val/${d}

for f in `ls ${d}/*.mp4 -alth|awk '{print $9}'|head -n 10`; do 

echo $f;

for MASK_RATIO in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
python3 run_videomae_vis.py \
    --mask_ratio ${MASK_RATIO} \
    --mask_type tube \
    --decoder_depth 4 \
    --model pretrain_videomae_base_patch16_224 \
    --img_path ${f} \
    --save_path ${SAVE_PATH} \
    --model_path ${MODEL_PATH}

done
done
done



fi

exit;

sbatch job_extract_feats_pretrain.sh 2000 10 0.4 4
sleep 1

sbatch job_extract_feats_pretrain.sh 2000 10 0.6 8
sleep 1

sbatch job_extract_feats_pretrain.sh 2000 10 0.9 16
sleep 1










