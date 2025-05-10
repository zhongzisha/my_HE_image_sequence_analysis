#!/bin/bash

#SBATCH --mail-type=FAIL

current_dir=`pwd`

if [ "$CLUSTER_NAME" == "FRCE" ]; then
    source $FRCE_DATA_ROOT/anaconda3/bin/activate th23
    module load cuda/11.8
    module load cudnn/8.8.3-cuda11
    if [ -d /tmp/zhongz2/data ]; then rm -rf /tmp/zhongz2/data; fi
    CACHE_ROOT=/tmp/zhongz2/$SLURM_JOB_ID
else
    source /data/zhongz2/anaconda3/bin/activate th24
    module load CUDA/12.1
    module load cuDNN/8.9.2/CUDA-12
    module load gcc/11.3.0
    CACHE_ROOT=/lscratch/$SLURM_JOB_ID
fi


srun python generate_videos_fromSeg.py \
--action "extract_patch_videos_original" \
--excel_filename mpp_025.xlsx \
--save_root /data/zhongz2/tcga_ffpe_all/patch_videos

exit;


mkdir -p /data/zhongz2/tcga_ffpe_all/patch_videos
sbatch --ntasks=512 --tasks-per-node=4 --partition=multinode --cpus-per-task=2 --time=108:00:00 --gres=lscratch:400 --mem=150G generate_videos_fromSeg.sh





















