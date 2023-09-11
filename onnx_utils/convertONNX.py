import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))
import cv2

import torchvision
from pathlib import Path
from models.hit_net_sf import HITNet_SF, HITNetXL_SF
from dataset.utils import np2torch
from colormap import apply_colormap, dxy_colormap
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse
from models import build_model

from predict import PredictModel


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HITNet_KITTI")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--output", default="./")
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = get_parser()

    model = PredictModel(**vars(args)).eval()
    ckpt = torch.load(args.ckpt)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.model.load_state_dict(ckpt)
    model1 = model.model
    model1.cuda()
    model1.eval()
    height = args.height
    width = args.width

    input_L = torch.randn(1, 3, height, width, device='cuda:0')
    input_R = torch.randn(1, 3, height, width, device='cuda:0')
    input_names = ['L', 'R']
    output_names = ['disp']
    # pred = model1(input_L, input_R)
    # print(pred)
    torch.onnx.export(
        model1,
        (input_L,input_R),
        args.output,
        verbose=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names)

    print("Finish!")
