




import sys,os,glob,shutil
import pickle
import numpy as np
import json
from matplotlib import pyplot as plt

params = [
    (0.4, 4),
    (0.6, 8),
    (0.9, 16)
]

result_dir = '/home/zhongz2/VideoMAEv2/OUTPUT/videomae2_vit_small_with_vit_base_teacher_k400_epoch_400/train2000_val_10'

fig, ax = plt.subplots()

for param in params:
    mask_ratio, batch_size = param
    with open(os.path.join(result_dir, f'mask{mask_ratio}_BS{batch_size}/log.txt'), 'r') as fp:
        lines = fp.readlines()
        data = [json.loads(line.strip()) for line in lines]
        xs = [item['epoch'] for item in data]
        ys = [item['train_loss'] for item in data]
        plt.plot(xs, ys, label='mask_ratio={}'.format(mask_ratio))

plt.xlabel('epoch')
plt.ylabel('train loss')
plt.grid()
plt.legend()
plt.savefig(os.path.join(result_dir, 'trainloss.png'))
plt.close('all')















