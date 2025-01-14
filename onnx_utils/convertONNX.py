import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))
import cv2
import torch
import argparse
from predict import PredictModel
from utils.file import MkdirSimple, ReadImageList
from onnxmodel import ONNXModel
import time
import numpy as np
from onnx_utils.onnx_test import test_dir

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HITNet_KITTI", choices=['HITNet_KITTI', 'HITNet_SF'])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--left_image", type=str, default="", help="test image left file or directory")
    parser.add_argument('--right_image', type=str, default="", help="test image right file or directory")
    parser.add_argument("--output", default="./")
    parser.add_argument("--test", action="store_true", help="test model")
    args = parser.parse_args()
    return args

def main():
    args = get_parser()

    model = PredictModel(**vars(args)).eval()
    ckpt = torch.load(args.ckpt)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.model.load_state_dict(ckpt)
    model = model.model
    model.cuda()
    model.eval()
    height = args.height
    width = args.width

    onnx_input = torch.randn(1, 3, height, width * 2, device='cuda:0')
    input_names = ['left']
    output_names = ['output']

    model_name = os.path.splitext(os.path.basename(args.ckpt))[0].replace(" ", "_")
    output = os.path.join(args.output, model_name, f'{args.width}_{args.height}')
    onnx_file = os.path.join(output, f'hitnet_{args.width}_{args.height}_{model_name}_12.onnx')
    MkdirSimple(output)
    torch.onnx.export(model,
                      onnx_input,
                      onnx_file,
                      # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=input_names,
                      output_names=output_names)

    print("export onnx to {}".format(onnx_file))
    if args.test:
        test_dir(onnx_file, [args.left_image, args.right_image], output)


if __name__ == '__main__':
    main()
