import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models import build_model

from predict import PredictModel

if __name__ == "__main__":
    import cv2
    import argparse
    import torchvision
    from pathlib import Path
    from models.hit_net_sf import HITNet_SF, HITNetXL_SF
    from dataset.utils import np2torch
    from colormap import apply_colormap, dxy_colormap

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs=2, required=True)
    parser.add_argument("--model", type=str, default="HITNet")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--output", default="./")
    args = parser.parse_args()

    model = PredictModel(**vars(args)).eval()
    ckpt = torch.load(args.ckpt)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.model.load_state_dict(ckpt)
    model1 = model.model
    model1.cuda()
    model1.eval()
    height = 400
    width = 640
    lp = Path(args.images[0])
    rp = Path(args.images[1])
    left = cv2.imread(str(lp), cv2.IMREAD_COLOR)
    right = cv2.imread(str(rp), cv2.IMREAD_COLOR)
    left = np2torch(left, bgr=True).cuda().unsqueeze(0)
    right = np2torch(right, bgr=True).cuda().unsqueeze(0)
    left = left * 2 - 1
    right = right * 2 - 1
    #
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
