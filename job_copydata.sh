#!/bin/bash


if [ "$CLUSTER_NAME" == "FRCE" ]; then
    cd /scratch/cluster_scratch/zhongz2/VideoMAEv2
else
    cd /home/zhongz2/VideoMAEv2
fi


set -x -e
echo "copy data on `hostname`"


IMG_KEY=${1}
NUM_TRAIN=${2}
NUM_VAL=${3}
SRC_DIR=${4}
DST_DIR=${5}


for subset in "train" "val"; do

mkdir -p ${DST_DIR}/${subset}

bash extract_tars.sh ${SRC_DIR}/${subset}_list_${IMG_KEY}_trn${NUM_TRAIN}_val${NUM_VAL}.txt ${SRC_DIR} ${DST_DIR}/${subset}

done


exit;










