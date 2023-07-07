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
    height = 640
    width = 960
    input_L = torch.randn(1, 3, height, width, device='cuda:0')
    input_R = torch.randn(1, 3, height, width, device='cuda:0')
    input_names = ['L', 'R']
    output_names = ['disp']
    # pred = model1(input_L, input_R)
    # print(pred)
    torch.onnx.export(
        model1,
        (input_L,input_R),
        "./hitnet_sf_finalpass_960_640_v12_lf_no_max_ceil.onnx",
        verbose=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names)

    print("Finish!")

    # model = build_model(args)
    # model.eval()
    # ckpt = torch.load(args.ckpt)
    # # for name in ckpt['state_dict']:
    # #     print('name is {}'.format(name))
    # model.load_state_dict(ckpt)
    # device = torch.device("cuda")
    # model = model.to(device)
    # input_names = ["input0"]  # ,"input1"
    # # output_names = ["output0"]
    # output_names = ["output_%d" % i for i in range(1)]
    # # output_names = ["output0","output1","output2","output3"]
    # print(output_names)
    # left = torch.randn(1, 3, 480, 640).to(device)
    # right = torch.randn(1, 3, 480, 640).to(device)
    # a = (left, right)
    # export_onnx_file = "./HITNet_SF_5.onnx"
    # # torch_out = torch.onnx._export(model(left,right),(left,right), output_onnx, export_params=True, verbose=False,
    # #                                input_names=input_names, output_names=output_names)
    # # torch_out = torch.onnx.export(model, args=(left, right), f=export_onnx_file, verbose=False, input_names=input_names,
    # #                               output_names=output_names, export_params=True, opset_version=12)
