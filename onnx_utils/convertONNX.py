#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: convert_onnx.py
@time: 2023/3/9 下午6:32
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

import argparse
import torch
from models.hit_net_sf import HITNet_SF
from models.hit_net_kitti import HITNet_KITTI

W, H = 640, 400

def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, help="model path")
    parser.add_argument("--output", type=str, help="output model path")
    parser.add_argument("--config", type=str, help="config yaml file")
    parser.add_argument("--type", type=str, default="sf", choices=['sf', 'kitti'], help="config yaml file")

    args = parser.parse_args()
    return args

def main():
    args = GetArgs()

    if 'sf' == args.type:
        model = HITNet_SF()
    else:
        model = HITNet_KITTI()
    model.eval()

    input_L = torch.randn(1, 3, H, W, device='cpu')
    input_R = torch.randn(1, 3, H, W, device='cpu')
    input_names = ['L']
    output_names = ['disp']
    # pred = model1(input_L, input_R)
    # print(pred)
    torch.onnx.export(
        model,
        input_L,
        args.output,
        export_params=True,  # store the trained parameter weights inside the model file
        verbose=True,
        opset_version=12,
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=input_names,
        output_names=output_names)


if __name__ == '__main__':
    main()
