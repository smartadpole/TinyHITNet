import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))
import cv2
import torch
import argparse
from predict import PredictModel
from tools.file import MkdirSimple, ReadImageList
from onnxmodel import ONNXModel
import time
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="HITNet_KITTI", choices=['HITNet_KITTI', 'HITNet_SF'])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--image", type=str, help="test image file or directory")
    parser.add_argument("--output", default="./")
    args = parser.parse_args()
    return args


def test_onnx(img_path, model, width=644, height=392):
    img_org = cv2.imread(img_path)
    img = cv2.resize(img_org, (width, height), cv2.INTER_LANCZOS4)
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    img = img / 255
    img = np.subtract(img, mean)
    img = np.divide(img, std)
    img = np.hstack([img, img])
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype("float32")

    start_time = time.time()
    output = model.forward(img)
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time) * 1000:.2f} ms")
    dis_array = output[0][0][0]
    dis_array = (dis_array - dis_array.min()) / (dis_array.max() - dis_array.min()) * 255.0
    dis_array = dis_array.astype("uint8")

    depth = cv2.resize(dis_array, (img_org.shape[1], img_org.shape[0]))
    depth = cv2.applyColorMap(cv2.convertScaleAbs(depth, 1), cv2.COLORMAP_PARULA)
    combined_img = np.vstack((img_org, depth))

    return combined_img, depth

def test_dir(image_dir, model_file, output_dir, width, height):
    model = ONNXModel(model_file)
    img_list = ReadImageList(image_dir)
    print("test image number: ", len(img_list))
    for file in img_list:
        image, depth = test_onnx(file, model, width, height)
        depth_file = os.path.join(output_dir, 'depth', os.path.basename(file))
        concat_file = os.path.join(output_dir, 'concat', os.path.basename(file))
        MkdirSimple(depth_file)
        MkdirSimple(concat_file)
        cv2.imwrite(concat_file, image)
        cv2.imwrite(depth_file, depth)

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

    MkdirSimple(args.output)
    onnx_file = os.path.join(args.output, os.path.splitext(os.path.basename(args.ckpt))[0] + ".onnx")
    torch.onnx.export(model,
                      onnx_input,
                      onnx_file,
                      # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=input_names,
                      output_names=output_names)

    test_dir(args.image, onnx_file, args.output, args.width, args.height)


if __name__ == '__main__':
    main()
