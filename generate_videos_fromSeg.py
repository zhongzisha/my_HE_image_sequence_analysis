




import sys,os,glob,shutil,pickle,json,argparse,io,tarfile,cv2,time,h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openslide
import hashlib
import idr_torch
# import pyvips
import tempfile
from skimage.transform import warp, AffineTransform
import multiprocessing
from PIL import Image, ImageFile, ImageDraw, ImageFilter, ImageFont
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True




def create_video_from_patches(patches,fps=10):
    """
    Creates a video from extracted patches.

    Parameters:
    - patches: List of image patches (assumed to be the same size).
    - fps: Frames per second for the output video.
    """
    if not patches:
        print("No patches to create a video.")
        return

    # Define video properties
    height, width = patches[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Common codec
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        out = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))
        # Write patches as frames
        for patch in patches:
            out.write(patch)
        out.release()

        with open(temp_file.name, 'rb') as f:
            video_bytes = f.read()
    os.remove(temp_file.name)

    return video_bytes

# from Yottixel code
def RGB2HSD(X): # Hue Saturation Density
    '''
    Function to convert RGB to HSD
    from https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization/blob/master/ops.py
    Args:
        X: RGB image
    Returns:
        X_HSD: HSD image
    '''
    eps = np.finfo(float).eps # Epsilon
    X[np.where(X==0.0)] = eps # Changing zeros with epsilon
    OD = -np.log(X / 1.0) # It seems to be calculating the Optical Density
    D  = np.mean(OD,3) # Getting density?
    D[np.where(D==0.0)] = eps # Changing zero densitites with epsilon
    cx = OD[:,:,:,0] / (D) - 1.0 
    cy = (OD[:,:,:,1]-OD[:,:,:,2]) / (np.sqrt(3.0)*D)
    D = np.expand_dims(D,3) # Hue?
    cx = np.expand_dims(cx,3) # Saturation
    cy = np.expand_dims(cy,3) # Density?
    X_HSD = np.concatenate((D,cx,cy),3)
    return X_HSD

def clean_thumbnail(thumbnail):
    '''
    Function to clean thumbnail
    Args:
        thumbnail: thumbnail image
    Returns:
        wthumbnail: cleaned thumbnail image
    '''
    # thumbnail array
    thumbnail_arr = np.asarray(thumbnail)
    # writable thumbnail
    wthumbnail = np.zeros_like(thumbnail_arr)
    wthumbnail[:, :, :] = thumbnail_arr[:, :, :]
    # Remove pen marking here
    # We are skipping this
    # This  section sets regoins with white spectrum as the backgroud regoin
    thumbnail_std = np.std(wthumbnail, axis=2)
    wthumbnail[thumbnail_std<5] = (np.ones((1,3), dtype="uint8")*255)
    thumbnail_HSD = RGB2HSD(np.array([wthumbnail.astype('float32')/255.]))[0]
    kernel = np.ones((30,30),np.float32)/900
    thumbnail_HSD_mean = cv2.filter2D(thumbnail_HSD[:,:,2],-1,kernel)
    wthumbnail[thumbnail_HSD_mean<0.05] = (np.ones((1,3),dtype="uint8")*255)
    # return writable thumbnail
    return wthumbnail



def largest_connected_component(binary_image):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    res = np.zeros(output.shape, dtype=np.uint8)
    res[output == max_label] = 1
    return res


def read_big_image(filename):
    slide = pyvips.Image.new_from_file(filename)
    img = np.ndarray(buffer=slide.write_to_memory(),
                        dtype=np.uint8,
                        shape=[slide.height, slide.width, slide.bands])
    return img

def write_big_image(filename, img):
    img_vips = pyvips.Image.new_from_array(img)
    img_vips.tiffsave(filename, compression="jpeg",
        tile=True, tile_width=512, tile_height=512,
        pyramid=True,  bigtiff=True)




def extract_rotated_patches(im, xc, yc, patch_size=224, step_size=8, num_frames=96, is_shown=False, save_root=None, svs_prefix=None, tar_fp_vid=None):

    H, W = im.shape[:2]

    cx, cy = W//2, H//2
    # cx, cy = coord

    # r = 1.5*patch_size//2
    r = patch_size//2
    rect0 = np.array([
        [cx-r, cy-r, 1],
        [cx-r, cy+r, 1],
        [cx+r, cy+r, 1],
        [cx+r, cy-r, 1]
    ])#.astype(np.int32)

    if is_shown:
        im1 = im.copy()
        cv2.circle(im1, (cx, cy), 3, (255, 0, 0), 3)

        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 255, 0)
        thickness = 2

    filenames = []
    for angle in [-180, -135, -90, -45, 0, 45, 90, 135]:

        # Get the rotation matrix
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        M1 = np.concatenate([M, np.array([[0, 0, 1]])], axis=0)

        patches = []        
        for step in np.arange(0, num_frames):
            rect = rect0.copy()
            rect[:, 0] += step*step_size            
            new_rect = np.dot(M1, rect.T)[:2, :].T# .astype(np.int32)    

            if is_shown: 
                cv2.drawContours(im1, [rect[:, :2].reshape(-1, 1, 2).astype(np.int32)], -1, (0, 255, 0), 1) 
                cv2.drawContours(im1, [new_rect[:, :2].reshape(-1, 1, 2).astype(np.int32)], -1, (0, 255, 255), 1) 
                cv2.putText(im1, str(angle), (int(new_rect[0,0]), int(new_rect[0,1])), fontFace, fontScale, color, thickness, cv2.LINE_AA)

            width = np.linalg.norm(new_rect[0] - new_rect[1])
            height = np.linalg.norm(new_rect[0] - new_rect[3])

            width, height = round(width), round(height)

            # if width != 2*r or height != 2*r:
            #     print(angle, width, height)

            pts_dst = np.array([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]], dtype=np.float32)

            M2 = cv2.getPerspectiveTransform(new_rect.astype(np.float32), pts_dst)

            warped_patch = cv2.warpPerspective(im, M2, (width, height))
            # warped_patch = warp(im, M2, output_shape=(width, height))
            # print(angle, warped_patch.shape)
            # w, h = warped_patch.shape[:2]
            # warped_patch = warped_patch[h//2-patch_size//2:h//2+patch_size//2, w//2-patch_size//2:w//2+patch_size//2, :]
            
            patches.append(warped_patch)
            # cv2.imwrite(f'/data/zhongz2/patch_{angle}.jpg', warped_patch)

        # print(angle, len(patches))
        if tar_fp_vid is not None:
            filename = '{}/x{}_y{}_{}.mp4'.format(svs_prefix, xc, yc, angle)
            video_bytes = create_video_from_patches(patches)
            v_buffer = io.BytesIO(video_bytes)
            info = tarfile.TarInfo(name=filename)
            info.size = v_buffer.getbuffer().nbytes
            info.mtime = time.time()
            v_buffer.seek(0)
            tar_fp_vid.addfile(info, v_buffer)
            filenames.append(filename)

    if is_shown and save_root is not None:
        cv2.imwrite('{}/{}_x{}_y{}_shown.jpg'.format(save_root, svs_prefix, xc, yc), im1)

    return filenames

def main(param):
    svs_filename, save_root = param
    debug = False
    if os.environ['CLUSTER_NAME'] == 'Biowulf':
        local_temp_dir = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], str(idr_torch.rank), str(idr_torch.local_rank))
    elif os.environ['CLUSTER_NAME'] == 'FRCE':
        local_temp_dir = os.path.join('/tmp/', os.environ['USER'], str(idr_torch.rank), str(idr_torch.local_rank))
    else:
        local_temp_dir = os.path.join(os.environ['HOME'], str(idr_torch.rank), str(idr_torch.local_rank))

    svs_filename1 = os.path.realpath(svs_filename)
    local_svs_filename = os.path.join(local_temp_dir, os.path.basename(svs_filename1))
    os.system(f'cp -RL "{svs_filename1}" "{local_svs_filename}"')
    time.sleep(1)


    svs_prefix = os.path.splitext(os.path.basename(svs_filename))[0]

    slide = openslide.open_slide(local_svs_filename)

    W, H = slide.level_dimensions[0]
    scale = 4000. / max(W, H)
    patch_size = 336
    size = int(patch_size*scale)

    thumbnail = slide.get_thumbnail((int(scale*W), int(scale*H)))
    if debug:
        thumbnail.save(f"{save_root}/{svs_prefix}_thumbnail.png")

    cthumbnail = clean_thumbnail(thumbnail)
    tissue_mask = (cthumbnail.mean(axis=2) != 255).astype(np.uint8)
    if debug:
        cv2.imwrite(f"{save_root}/{svs_prefix}_tissue_mask.png", tissue_mask*255)

    im_floodfill = tissue_mask.copy()
    h, w = tissue_mask.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    tissue_mask_filled = tissue_mask | im_floodfill_inv
    if debug:
        cv2.imwrite(f"{save_root}/{svs_prefix}_tissue_mask_filled.png", tissue_mask_filled)

    tissue_mask = (tissue_mask_filled / 255).astype(np.uint8)
    # for erosion_size in [5, 15, 25, 35, 45, 55, 75, 85, 95]:
    if True:
        erosion_size = 115
        erosion_shape = cv2.MORPH_ERODE
        element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))

        tissue_mask1 = cv2.erode(tissue_mask, element)     
        if debug:
            cv2.imwrite(f"{save_root}/{svs_prefix}_tissue_mask_erode{erosion_size}.png", tissue_mask1*255)

    try:
        tissue_mask = largest_connected_component(tissue_mask1)  
        if debug: 
            cv2.imwrite(f"{save_root}/{svs_prefix}_tissue_mask_erode_refined.png", tissue_mask*255)  
    except:
        print('error in largest_connected_component: ', svs_prefix)
        return

    if np.count_nonzero(tissue_mask) < 20:
        print('no enough points, ignore: ', svs_prefix)
        return


    # patch_size = 224
    step_size = 8
    num_frames = 96
    # is_shown=False
    # save_root=os.path.join('/lscratch', os.environ['SLURM_JOB_ID'])
    # prefix=os.path.splitext(os.path.basename(slide_path))[0]
    max_r = num_frames*step_size+patch_size

    yx = np.stack(np.where(tissue_mask)).T/scale
    ymin,xmin = yx.min(axis=0)
    ymax,xmax = yx.max(axis=0)

    if abs(xmax-xmin)<2.5*max_r or abs(ymax-ymin)<2.5*max_r:
        print('small tissues, ignore: ', svs_prefix)
        return

    # cxs, cys = np.meshgrid(np.arange(xmin, xmax, patch_size//2), np.arange(ymin, ymax, patch_size//2))
    cxs, cys = np.meshgrid(np.arange(xmin-max_r//2, xmax+max_r//2, max_r), np.arange(ymin-max_r//2, ymax+max_r//2, max_r))

    cxs = cxs.flatten()
    cys = cys.flatten()
    cxy = np.stack([cxs, cys]).T.astype(np.int32)

    patch_size1 = int(max_r * scale)
    cxy1 = (cxy*scale).astype(np.int32)

    if debug:
        he_shown1 = np.array(thumbnail.convert('RGB')).copy()
        for cx,cy in cxy1: #zip(cxs, cys):
            x1,x2 = cx-patch_size1//2, cx+patch_size1//2
            y1,y2 = cy-patch_size1//2, cy+patch_size1//2

            cross_color = (0, 0, 255)
            cross_thickness = 5
            cross_length = 11
            center_x, center_y = int(cx), int(cy)
            
            if tissue_mask[center_y, center_x] > 0:
                cv2.rectangle(he_shown1, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 8)
                cv2.line(he_shown1, (center_x - cross_length // 2, center_y), (center_x + cross_length // 2, center_y), cross_color, cross_thickness)
                cv2.line(he_shown1, (center_x, center_y - cross_length // 2), (center_x, center_y + cross_length // 2), cross_color, cross_thickness)

        cv2.imwrite(f"{save_root}/{svs_prefix}_tissue_mask_erode_refined_shown.png", he_shown1)  
        time.sleep(1) 


    valid_inds = []
    for ii, (xc1,yc1) in enumerate(cxy1): 
        if tissue_mask[int(yc1), int(xc1)] > 0:
            valid_inds.append(ii)

    fh_vid = io.BytesIO()
    tar_fp_vid = tarfile.open(fileobj=fh_vid, mode='w:gz')

    select_count = 128
    if len(valid_inds) > select_count:
        inds = np.random.choice(valid_inds, size=select_count, replace=False)
        cxy = cxy[inds]
        cxy1 = cxy1[inds]

    # count = 0
    l = len(cxy)
    all_filenames = []
    for ii, ((xc,yc),(xc1,yc1)) in enumerate(zip(cxy,cxy1)):

        # if debug and count == 5:
        #     break

        # if tissue_mask[int(yc1), int(xc1)] == 0:
        #     continue

        im = np.array(slide.read_region((xc-max_r, yc-max_r),0,(2*max_r, 2*max_r)).convert('RGB'))[:,:,::-1]

        # extract_rotated_patches(im, xc, yc, patch_size=patch_size, step_size=step_size, \
        #     num_frames=num_frames, svs_prefix=svs_prefix, tar_fp_vid=tar_fp_vid,\
        #         is_shown=True, save_root=save_root) 
        if debug:
            filenames = extract_rotated_patches(im, xc, yc, patch_size=patch_size, step_size=step_size, \
                num_frames=num_frames, svs_prefix=svs_prefix, tar_fp_vid=tar_fp_vid, \
                    is_shown=True, save_root=save_root)
        else:
            filenames = extract_rotated_patches(im, xc, yc, patch_size=patch_size, step_size=step_size, \
                num_frames=num_frames, svs_prefix=svs_prefix, tar_fp_vid=tar_fp_vid)
        all_filenames.extend(filenames)
        # print(svs_prefix, l, ii)
        # print(xc,yc)
        # count += 1

    slide.close()

    tar_fp_vid.close()
    with open('{}/{}.tar.gz'.format(save_root, svs_prefix), 'wb') as fp:
        fp.write(fh_vid.getvalue())
    if len(all_filenames) > 0:
        with open('{}/{}.txt'.format(save_root, svs_prefix), 'w') as fp:
            fp.write('\n'.join(all_filenames))


    time.sleep(1)
    if os.path.exists(local_svs_filename):
        os.system(f'rm -rf "{local_svs_filename}"')
        print(f'removed {local_svs_filename}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, default="")
    parser.add_argument("--excel_filename", type=str, default="")
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--save_root", type=str, default="")
    parser.add_argument("--patch_size", type=int, default=2048)
    parser.add_argument("--scale_list", type=str, default="0.75,0.5,0.25")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--crop_step", type=int, default=256)
    parser.add_argument("--thresh_size", type=int, default=200)
    return parser.parse_args()



def extract_patch_videos_original(args): # no scale and crop

    df = pd.read_excel(args.excel_filename) if 'xlsx' in args.excel_filename else pd.read_csv(args.excel_filename)
    
    indices = np.arange(len(df))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size) 
    sub_df = df.iloc[index_splits[idr_torch.rank]]
    sub_df = sub_df.reset_index(drop=True)

    # for ind, row in sub_df.iterrows():
    #     main(row['image_filename'], debug=False, save_root=args.save_root)


    if os.environ['CLUSTER_NAME'] == 'Biowulf':
        local_temp_dir = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], str(idr_torch.rank), str(idr_torch.local_rank))
    elif os.environ['CLUSTER_NAME'] == 'FRCE':
        local_temp_dir = os.path.join('/tmp/', os.environ['USER'], str(idr_torch.rank), str(idr_torch.local_rank))
    else:
        local_temp_dir = os.path.join(os.environ['HOME'], str(idr_torch.rank), str(idr_torch.local_rank))
    os.makedirs(local_temp_dir, exist_ok=True)


    with multiprocessing.Pool(processes=8) as pool:
        pool.map(main, [(f, args.save_root) for f in sub_df['image_filename'].values])



if __name__ == '__main__':
    args = get_args()

    if args.action == "extract_patch_videos_original":
        extract_patch_videos_original(args)
    else:
        print('wrong action')










