import glob
import os
import numpy as np
import cv2
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import random

SEED = 42

CLASSES = ('background', 'building', 'road', 'water', 'barren', 'forest',
           'agricultural')

PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
           [159, 129, 183], [0, 255, 0], [255, 195, 128]]


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", default="data/LoveDA/Train/Rural/masks_png")
    parser.add_argument("--output-mask-dir", default="data/LoveDA/Train/Rural/masks_png_convert")
    parser.add_argument("--rgb2mask", action='store_true', help='Chose convert rgb to mask mode')
    return parser.parse_args()


def convert_label(mask):
    # mask[mask == 0] = 8
    # mask -= 1

    return mask

def rgb2label(rgb_label):
    palette = {        
        0: [255, 255, 255],
        1: [0,0,255],
        2: [0,255,255],
        3: [255,0,0],
        4: [183, 129, 159],
        5: [0,255,0],
        6: [128, 195, 255],
    }
    img_classes = np.full_like(rgb_label[:, :, 0], 255, dtype=np.uint8)
    for label, rgb in palette.items():
        img_classes[(rgb_label==rgb).all(axis=2)] = label   
    return img_classes 

def patch_format_rgb2label(inp):
    (mask_path, masks_output_dir) = inp
    # print(mask_path, masks_output_dir)
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    # label = convert_label(mask)
    
    label = rgb2label(mask.copy())
    # print(np.unique(label))
    # rgb_label = cv2.cvtColor(rgb_label, cv2.COLOR_RGB2BGR)
    out_mask_path_rgb = os.path.join(masks_output_dir , "{}.png".format(mask_filename))
    cv2.imwrite(out_mask_path_rgb, label)

    # out_mask_path = os.path.join(masks_output_dir, "{}.png".format(mask_filename))
    # cv2.imwrite(out_mask_path, label)

def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 0, 255]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [159, 129, 183]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 195, 128]
    return mask_rgb




def patch_format_label2rgb(inp):
    (mask_path, masks_output_dir) = inp
    # print(mask_path, masks_output_dir)
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    # mask[mask==0]=255
    mask = mask
    print(np.unique(mask))
    # label = convert_label(mask)

    rgb_label = label2rgb(mask.copy())
    rgb_label = cv2.cvtColor(rgb_label, cv2.COLOR_RGB2BGR)
    # print(rgb_label.shape)
    out_mask_path_rgb = os.path.join(masks_output_dir + '_rgb', "{}.png".format(mask_filename))
    cv2.imwrite(out_mask_path_rgb, rgb_label)

    # out_mask_path = os.path.join(masks_output_dir, "{}.png".format(mask_filename))
    # cv2.imwrite(out_mask_path, label)

import zipfile
def zip_folder(folder_path):
    # Ensure the folder path is absolute
    folder_path = os.path.abspath(folder_path)

    # Get the folder name and use it for the zip file name
    folder_name = os.path.basename(folder_path.rstrip('/\\'))
    zip_filename = f"{folder_name}.zip"
    zip_filepath = os.path.join(os.path.dirname(folder_path), zip_filename)

    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Keep the relative folder structure
                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname)

    print(f"Zipped folder to: {zip_filepath}")

if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    masks_dir = args.mask_dir
    if args.rgb2mask:
        masks_output_dir=args.mask_dir+"_masks"
        print("Masks output dir: ", masks_output_dir)
    else:
        masks_output_dir = args.mask_dir
    mask_paths = glob.glob(os.path.join(masks_dir, "*.png"))

    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
        # os.makedirs(masks_output_dir + '_rgb')

    inp = [(mask_path, masks_output_dir) for mask_path in mask_paths]

    t0 = time.time()
    if args.rgb2mask:
        mpp.Pool(processes=mp.cpu_count()).map(patch_format_rgb2label, inp)
    else:
        mpp.Pool(processes=mp.cpu_count()).map(patch_format_label2rgb, inp)        
    t1 = time.time()
    split_time = t1 - t0
    print('Images converting spends: {} s'.format(split_time))
    zip_folder(masks_output_dir)

