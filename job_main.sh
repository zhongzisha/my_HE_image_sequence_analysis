#!/bin/bash

#SBATCH --job-name=debug2
#SBATCh --mail-type=FAIL
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100x:4,lscratch:1000
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

export OMP_NUM_THREADS=1 
if [ "${SLURM_JOB_NODELIST}" != "" ]; then
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export NNODES=$SLURM_NNODES 
else
    export MASTER_ADDR=`hostname`
    export NNODES=1 
fi
export MASTER_PORT=$((12000 + $RANDOM % 20000))

export NUM_TRAIN=2000 #${1}
export NUM_VAL=10  #${2}
export MASK_RATIO=0.4
export BATCH_SIZE=4

export DST_DIR=${MYTMP_DIR}/videomaev2_${NUM_TRAIN}_${NUM_VAL}
rm -rf ${DST_DIR}/*

mkdir -p ${DST_DIR}
python generate_train_val.py ${NUM_TRAIN} ${NUM_VAL} ${SRC_DIR} ${SRC_DIR}
sleep 1

if [ $NNODES -eq 1 ]; then
    bash job_copydata.sh video ${NUM_TRAIN} ${NUM_VAL} ${SRC_DIR} ${DST_DIR}
else
    srun --export ALL --jobid $SLURM_JOB_ID bash job_copydata.sh video ${NUM_TRAIN} ${NUM_VAL} ${SRC_DIR} ${DST_DIR}
    wait
fi

if [ $NNODES -eq 1 ]; then
    bash job_pretrain.sh
else
    srun --export ALL --jobid $SLURM_JOB_ID bash job_pretrain.sh
    wait
fi

exit;


sbatch job_main.sh 1000 10


# for validation
NUM_TRAIN=2000
NUM_VAL=10
BATCH_SIZE=4
MASK_RATIO=0.4
subset="val"
IMG_KEY="video"
export DST_DIR=${MYTMP_DIR}/videomaev2_${NUM_TRAIN}_${NUM_VAL}
mkdir -p ${DST_DIR}
bash extract_tars.sh ${SRC_DIR}/${subset}_list_${IMG_KEY}_trn${NUM_TRAIN}_val${NUM_VAL}.txt ${SRC_DIR} ${DST_DIR}/${subset}

d=$DST_DIR/val/TCGA-25-1312-01Z-00-DX1.733EC7A7-0FC8-4DDC-B366-DF5A45D6BB4E
d=${DST_DIR}/val/TCGA-DX-AB2H-01Z-00-DX1.F595B9EE-062B-41CA-B96C-58E8F11963AF
# path to pretrain model
MODEL_PATH='/scratch/cluster_scratch/zhongz2/VideoMAEv2/OUTPUT/pretrain/checkpoint-199.pth'
MODEL_PATH=OUTPUT/videomae2_vit_small_with_vit_base_teacher_k400_epoch_400/train${NUM_TRAIN}_val_${NUM_VAL}/mask${MASK_RATIO}_BS${BATCH_SIZE}/checkpoint-15.pth
MODEL_PATH=`ls OUTPUT/videomae2_vit_small_with_vit_base_teacher_k400_epoch_400/train${NUM_TRAIN}_val_${NUM_VAL}/mask${MASK_RATIO}_BS${BATCH_SIZE}/checkpoint*.pth -alth | head -n 1 | awk '{ print $9 }'`
# Set the path to save video
OUTPUT_DIR=/data/zhongz2/VideoMAEv2_results/${MODEL_PATH}
mkdir -p $OUTPUT_DIR;

for f in `find ${d} -name "*.mp4"`; do 

echo $f;

for MASK_RATIO in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
python3 run_videomae_vis.py \
    --mask_ratio ${MASK_RATIO} \
    --mask_type tube \
    --decoder_depth 4 \
    --model pretrain_videomae_base_patch16_224 \
    --img_path ${f} \
    --save_path ${OUTPUT_DIR}/$(basename ${f})/mask_ratio=${MASK_RATIO} \
    --model_path ${MODEL_PATH}
done

break

done

