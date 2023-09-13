import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

import cv2
import torch
import tqdm
import numpy as np
import HitnetModule
from pathlib import Path
import multiprocessing as mp

from dataset.utils import np2torch

import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, required=True, help="data dir for disp image")
    parser.add_argument("--middle_path", type=str, required=True, help="middle data dir for combine image_dir with image_list")
    parser.add_argument("--image_list", type=str, required=True, help="disp image path after middle")
    parser.add_argument("--middle_save", type=str, required=True, help="new middle path for save slant image")

    return parser.parse_args()

def process(file_path):
    while True:
        for ids, lock in enumerate(process.lock_list):
            if lock.acquire(block=False):
                disp_path = (process.root / process.middle_path / file_path).with_suffix(
                    ".png"
                )
                dxy_path = (process.root / process.middle_save / file_path).with_suffix(
                    ".npy"
                )
                dxy_path.parent.mkdir(exist_ok=True, parents=True)
                with torch.no_grad():
                    x = (
                        np2torch(
                            cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED).astype(
                                np.float32
                            )
                            / 256
                        )
                        .unsqueeze(0)
                        .cuda(ids)
                    )
                    x = HitnetModule.plane_fitting(x, 1024, 1, 9, 1e-3, 1e5)
                    x = x[0].cpu().numpy()
                np.save(dxy_path, x)
                lock.release()
                return


def process_init(lock_list, root, middle_path, middle_save):
    process.lock_list = lock_list
    process.root = root
    process.middle_path = middle_path
    process.middle_save = middle_save


def main(root, list_path, middle_path, middle_save):
    root = Path(root)

    with open(list_path, "rt") as fp:
        file_list = [Path(line.strip()) for line in fp]

    lock_list = [mp.Lock() for _ in range(1)]
    with mp.Pool(1, process_init, [lock_list, root, middle_path, middle_save]) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process, file_list), total=len(file_list)))


if __name__ == "__main__":
    args = get_parser()
    # main("/data/ABBY/DEPTH/TRAIN/data_Capture_Img_Tof", "/data/ABBY/DEPTH/TRAIN/data_Capture_Img_Tof/lists/indemind_09_12.txt")
    main(args.image_dir, args.image_list, args.middle_path, args.middle_save)
