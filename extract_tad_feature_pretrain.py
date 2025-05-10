"""Extract features for temporal action detection datasets"""
import argparse
import os,glob
import random
import pickle

import numpy as np
import torch
from timm.models import create_model
from torchvision import transforms
from packaging import version
# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
from dataset.loader import get_video_loader
import idr_torch

def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def resize(vid, size, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid,
        size=size,
        scale_factor=scale,
        mode=interpolation,
        align_corners=False)


class ToFloatTensorInZeroOne(object):

    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)


def get_args():
    parser = argparse.ArgumentParser(
        'Extract TAD features using the videomae model', add_help=False)

    parser.add_argument(
        '--action',
        default='step1',
        type=str,
        help='which action'
    )

    parser.add_argument(
        '--data_set',
        default='HE_video',
        choices=['THUMOS14', 'FINEACTION', 'HE_video'],
        type=str,
        help='dataset')

    parser.add_argument(
        '--data_path',
        # default='/mnt/gridftp/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version8_with_video/rot+000/448/10000/rot0_sub',
        default='/tmp/zhongz2/448/10000/rot0',
        type=str,
        help='dataset path')
    parser.add_argument(
        '--save_path',
        # default='/mnt/gridftp/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version8_with_video/rot+000/448/10000/rot0_sub_feats',
        default='/tmp/zhongz2/448/10000/rot0_results',
        type=str,
        help='path for saving features')

    parser.add_argument(
        '--model',
        default='vit_base_patch16_224',
        type=str,
        metavar='MODEL',
        help='Name of model')
    parser.add_argument(
        '--ckpt_path',
        default='/scratch/cluster_scratch/zhongz2/VideoMAEv2/OUTPUT/finetune/checkpoint-49.pth',
        help='load from checkpoint')

    return parser.parse_args()


def get_start_idx_range(data_set):

    def thumos14_range(num_frames):
        return range(0, num_frames - 15, 4)

    def fineaction_range(num_frames):
        return range(0, num_frames - 15, 16)

    if data_set == 'THUMOS14':
        return thumos14_range
    elif data_set == 'FINEACTION':
        return fineaction_range
    elif data_set == 'HE_video':
        return thumos14_range
    else:
        raise NotImplementedError()


def extract_feature(args):
    # preparation

    data_root = '/data/zhongz2/tcga_ffpe_all/patch_videos/'
    save_root = '/data/zhongz2/tcga_ffpe_all/patch_videos_features/'
    save_root = args.ckpt_path.replace('.pth', '_results')
    if idr_torch.rank == 0:
        os.makedirs(save_root, exist_ok=True)
    with open('/data/zhongz2/tcga_ffpe_all/patch_videos/val_list_video_trn2000_val10.txt', 'r') as fp:
        lines = fp.readlines()

    indices = np.arange(len(lines))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size)
    print('local rank:', idr_torch.local_rank)
    print('world_size: ', idr_torch.world_size)
    sub_lines = [lines[i] for i in index_splits[idr_torch.rank]] # df.iloc[index_splits[idr_torch.rank]]

    video_loader = get_video_loader()
    start_idx_range = get_start_idx_range(args.data_set)
    transform = transforms.Compose(
        [ToFloatTensorInZeroOne(),
         Resize((224, 224))])

    model = create_model(
        'pretrain_videomae_base_patch16_224',
        pretrained=False,
        drop_path_rate=0., #args.drop_path,
        drop_block_rate=None,
        all_frames=16, #args.num_frames,
        tubelet_size=2, #args.tubelet_size,
        decoder_depth=4, #args.decoder_depth,
        with_cp=False #args.with_checkpoint
        )

    if version.parse(torch.__version__) > version.parse('1.13.1'):
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    # ckpt = torch.load('/home/zhongz2/VideoMAEv2/OUTPUT/videomae2_vit_small_with_vit_base_teacher_k400_epoch_400/train2000_val_10/mask0.6_BS8/checkpoint-6.pth', map_location='cpu', weights_only=False)

    ckpt = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    for model_key in ['model', 'module']:
        if model_key in ckpt:
            ckpt = ckpt[model_key]
            break
    model.load_state_dict(ckpt)
    model.eval()
    model.cuda()

    local_temp_dir = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], str(idr_torch.rank), str(idr_torch.local_rank))
    os.makedirs(local_temp_dir, exist_ok=True)

    # decompress the tarfiles
    for line in sub_lines:
        line = line.strip()
        os.system('tar -xf "{}" -C "{}"'.format(os.path.join(data_root, line), local_temp_dir))

    # get video path
    vid_list = glob.glob('{}/**/*.mp4'.format(local_temp_dir))
    # random.shuffle(vid_list)

    # extract feature
    num_videos = len(vid_list)
    invalid_video_names = []
    alldata = []
    for idx, video_path in enumerate(vid_list):

        try:
            vr = video_loader(video_path)
            num_frames = len(vr)
            print(video_path, num_frames)
        except:
            invalid_video_names.append(video_path)
            continue
        
        # if num_frames != 76:
        #     invalid_video_names.append(video_path)
        #     continue

        feature_list = []
        if True: #for start_idx in start_idx_range(len(vr)):
            # data = vr.get_batch(np.arange(start_idx, start_idx + 16)).asnumpy()
            data = vr.get_batch(np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60])).asnumpy()
            frame = torch.from_numpy(data)  # torch.Size([16, 566, 320, 3])
            frame_q = transform(frame)  # torch.Size([3, 16, 224, 224])
            input_data = frame_q.unsqueeze(0).cuda()

            with torch.no_grad():
                feature = model.forward_features(input_data)
                feature = feature.cpu().numpy()

        # [N, C]
        # np.save(url, np.vstack(feature_list))
        # print(f'[{idx} / {num_videos}]: save feature on {url}')
        alldata.append((video_path, feature))

        # if idx == 5:
        #     break

    prefix = 'part_{}_{}'.format(idr_torch.rank, idr_torch.local_rank)
    with open(os.path.join(save_root, f'{prefix}_invalid.pkl'), 'wb') as fp:
        pickle.dump({'invalid_video_names': invalid_video_names}, fp)

    with open(os.path.join(save_root, f'{prefix}_alldata.pkl'), 'wb') as fp:
        pickle.dump({'alldata': alldata}, fp)




def plot_umap():

    import numpy as np
    import umap

    import glob
    import pickle
    import os

    files = glob.glob('/data/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version8_with_video_noRandom/*/alldata.pkl')

    all_feats = []
    for f in files:
        with open(f, 'rb') as fp:
            alldata = pickle.load(fp)['alldata']
        all_feats.extend(alldata)

    all_filenames = [v[0] for v in all_feats]
    all_feats_data = np.concatenate([v[1] for v in all_feats])
    all_labels = [os.path.basename(f).split('_')[0] for f in all_filenames]
    # labels_dict = {label:ind for ind,label in enumerate(np.unique(all_labels))}
    labels_dict = {f'rot{v}': ind for ind, v in enumerate([-180, -135, -90, -45, 0, 45, 90, 135, 180])}
    all_labels = [labels_dict[v] for v in all_labels]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_feats_data)
    reducer = umap.UMAP(random_state=42, metric="euclidean", n_components=3)
    feats_embedding = reducer.fit_transform(X_scaled)

def plot_umap3d_2():
    import sklearn.datasets
    import pandas as pd
    import pickle
    import numpy as np
    import umap
    import umap.plot
    with open('alldata_umap3d.pkl', 'rb') as fp:
        data = pickle.load(fp)
    feats_embedding = data['feats_embedding']
    all_labels = np.array(data['all_labels'])
    X_scaled = data['X_scaled']

    pendigits = sklearn.datasets.load_digits()
    mnist = sklearn.datasets.fetch_openml('mnist_784')
    fmnist = sklearn.datasets.fetch_openml('Fashion-MNIST')

    hover_data = pd.DataFrame({'index':np.arange(len(feats_embedding)),
                            'label':all_labels})

    labels_dict = {ind: f'rot{v}' for ind, v in enumerate([-180, -135, -90, -45, 0, 45, 90, 135, 180])}

    hover_data['item'] = hover_data.label.map(
        labels_dict
    )

    mapper = umap.UMAP(random_state=42, metric="euclidean", n_components=2)
    feats_embedding = mapper.fit(X_scaled)

    p = umap.plot.interactive(mapper, labels=all_labels, hover_data=hover_data, point_size=2)
    umap.plot.show(p)


    from matplotlib import pyplot as plt
    feats_embedding = data['feats_embedding'].copy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Create the scatter plot
    ax.scatter(feats_embedding[:,0],feats_embedding[:,1], feats_embedding[:,2])

    # Add labels to each point
    for i, label in enumerate(labels):
        ax.text(x[i], y[i], z[i], label)


def step2_umap_for_pretrain_features(args):

    import sys,os,pickle
    import glob
    import numpy as np

    save_root = args.ckpt_path.replace('.pth', '_results')

    files = glob.glob(f'{save_root}/*_alldata.pkl')

    all_filenames = []
    all_feats = []
    for f in files:
        with open(f, 'rb') as fp:
            data = pickle.load(fp)
        for item in data['alldata']:
            all_filenames.append(item[0])
            all_feats.append(item[1])

    all_feats = np.concatenate(all_feats) # Nx768

    all_feats /= np.linalg.norm(all_feats, axis=1)[:, None] # normalized to unit 1

    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances

    n_clusters = 16
    model = KMeans(n_clusters=n_clusters)
    model.fit(all_feats)

    kmeans_centers = model.cluster_centers_
    kmeans_labels = model.labels_

    topn = 10

    # extract images
    data_root = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'videomaev2_2000_10/val/')
    video_loader = get_video_loader()

    # save_root = '/data/zhongz2/tcga_ffpe_all/patch_videos_features/'
    import cv2
    for ci, c in enumerate(np.unique(kmeans_labels)):
        inds = np.where(kmeans_labels == c)[0]
        feats = all_feats[inds,:]
        dists = pairwise_distances(kmeans_centers[ci, :][None], feats)[0]
        sort_inds = np.argsort(dists)[:topn]  # 升序
        selected_filenames = [all_filenames[inds[si]] for si in sort_inds]

        all_images = []
        for filename in selected_filenames:
            splits = filename.split('/')
            video_path = os.path.join(data_root, splits[-2], splits[-1])
            vr = video_loader(video_path)
            data = vr.get_batch(np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60])).asnumpy()

            all_images.append(data.reshape(-1, 336, 3))
        
        all_images1 = np.concatenate(all_images, axis=1)[:,:,::-1].transpose((1, 0, 2))
        cv2.imwrite(os.path.join(save_root, f'cluster{ci}.jpg'), all_images1)
        

    # import umap
    # import umap.plot

    # mapper = umap.UMAP(random_state=42, metric="cosine", n_components=2)
    # feats_embedding = mapper.fit(kmeans_centers)


if __name__ == '__main__':
    args = get_args()
    if args.action == 'step1':
        extract_feature(args)
    elif args.action == 'step2':
        step2_umap_for_pretrain_features(args)
