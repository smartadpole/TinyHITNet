import onnxruntime
import os
import sys
import torch
import torchvision

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

from onnxmodel import ONNXModel
import cv2
from dataset.utils import np2torch, np2float
import numpy as np
import cv2
import argparse
import torchvision
from pathlib import Path
from models.hit_net_sf import HITNet_SF, HITNetXL_SF
from dataset.utils import np2torch
from colormap import apply_colormap, dxy_colormap

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_file", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_parser()

    model = ONNXModel(args.onnx_file)
    lp = args.left # "/data/ljdong/Depth_Estimate/image/KITTI_size_1242X375/left/000000_10.png"
    rp = args.right #"/data/ljdong/Depth_Estimate/image/KITTI_size_1242X375/right/000000_10.png"

    # model = ONNXModel("ONNX_SAVE/hitnet_kitti_640_400_const1.onnx")
    # lp = "/data/ljdong/Depth_Estimate/image/left/0000000000.png"
    # rp = "/data/ljdong/Depth_Estimate/image/right/0000000000.png"

    left = cv2.imread(str(lp), cv2.IMREAD_COLOR)
    right = cv2.imread(str(rp), cv2.IMREAD_COLOR)

    left = np2float(left, True)
    right = np2float(right, True)

    left = np.expand_dims(left, axis=0).astype(np.float32)
    right = np.expand_dims(right, axis=0).astype(np.float32)
    left_write = left.copy()
    left = left * 2 - 1
    right = right * 2 - 1

    print(left.shape)
    # output = model.forward({"L":left, "R":right})
    output = model.forward2((left, right))

    disp = output[0][:, 0:1]
    print(len(output), output[0].shape)
    disp = torch.from_numpy(disp)
    disp = np.clip(disp / 192 * 255, 0, 255).long()

    disp = apply_colormap(disp)
    output1 = [torch.from_numpy(left_write), disp]

    output1 = torch.cat(output1, dim=0)
    torchvision.utils.save_image(output1, args.output_file, nrow=1)
    cv2.imwrite("111.png", output1)