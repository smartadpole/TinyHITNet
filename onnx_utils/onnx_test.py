#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: sunhao
@contact: smartadpole@163.com
@file: onnx_test.py
@time: 2025/1/8 15:15
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))
import argparse
import cv2
import numpy as np
from onnxmodel import ONNXModel
from utils.file import MkdirSimple, match_stereo_file, ReadImageList
import os
import time

DEFAULT_COUNT = 10
MAX_COUNT = 500
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]

def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="model path")
    parser.add_argument("-o", "--output", type=str, required=True, help="output model path")
    parser.add_argument("--left_image", type=str, default="", help="test image left file or directory")
    parser.add_argument('--right_image', type=str, default="", help="test image right file or directory")

    args = parser.parse_args()
    return args

def resize_padding_right_top(img, size):
    h, w = img.shape[:2]
    target_h, target_w = size

    # Calculate the scaling factor to maintain aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image
    img = cv2.resize(img, (new_w, new_h))

    # Calculate padding
    pad_h = target_h - new_h
    pad_w = target_w - new_w

    # Pad the image
    padded_img = cv2.copyMakeBorder(img, pad_h, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    return padded_img

def inverse_resize_padding_right_top(img, origin_size):
    h, w = img.shape[:2]
    origin_h, origin_w = origin_size
    scale = min(w / origin_w, h / origin_h)
    new_w = int(origin_w * scale)
    new_h = int(origin_h * scale)

    # Remove padding
    img = img[h - new_h:, :new_w]

    # Resize back to original size
    img = cv2.resize(img, (w, h))
    return img

def resize(img, size):
    target_h, target_w = size
    img = cv2.resize(img, (target_w, target_h))

    return img

def inverse_resize(img, origin_size):
    origin_h, origin_w = origin_size
    img = cv2.resize(img, (origin_w, origin_h))
    return img

def transpose_image(img):
    return np.transpose(img, (2, 0, 1))

def to_tensor(img):
    return img / 255.0

def normalize_image(img, mean, std):
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    img = (img - mean) / std
    return img

def inference(imgs, model):
    c, h, w = model.get_input_size()
    w = w // len(imgs)

    tensors = []
    for img in imgs:
        if c == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = resize(img, [h, w])
        img = transpose_image(img)
        img = to_tensor(img)
        img = normalize_image(img, mean, std)
        tensors.append(img)

    tensors = np.concatenate(tensors, axis=2)
    tensors = np.expand_dims(tensors, axis=0).astype("float32")

    start_time = time.time()
    output = model.forward(tensors)
    end_time = time.time()
    print(f"Execution time: {(end_time - start_time) * 1000:.2f} ms")
    dis_array = np.squeeze(output)

    return dis_array

def visual_image(img, depth):
    depth = cv2.resize(depth, (img.shape[1], img.shape[0]))
    depth = cv2.applyColorMap(cv2.convertScaleAbs(depth, 1), cv2.COLORMAP_HSV)
    combined_img = np.hstack((img, depth)) if img.shape[1] < img.shape[0] else np.vstack((img, depth))

    return combined_img

def test_image(image_files: list, model):
    if not all(os.path.basename(image_files[0]) == os.path.basename(f) for f in image_files):
        print("All file names in image_files must be the same")
        return

    c, h, w = model.get_input_size()
    imgs = []

    for image_file in image_files:
        if image_file == "":
            img = np.clip(np.random.rand(h, w, c) * 255, 0, 255).astype("float32")
        else:
            img = cv2.imread(image_file)

        imgs.append(img)

    img_org = imgs[0]
    disp = inference(imgs, model)
    disp = inverse_resize(disp, img_org.shape[:2])

    disp_norm = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_norm = disp_norm.astype("uint8")

    combined_img = visual_image(img_org, disp_norm)

    return combined_img, disp, disp_norm

def save_image(image, depth, output_dir, file_name):
    depth_file = os.path.join(output_dir, 'depth', file_name)
    concat_file = os.path.join(output_dir, 'concat', file_name)
    MkdirSimple(depth_file)
    MkdirSimple(concat_file)
    cv2.imwrite(concat_file, image)
    cv2.imwrite(depth_file, depth)

def test_dir(model_file, image_dirs, output_dir):
    model = ONNXModel(model_file)

    print(f"model file: {model_file}")
    print(f"dataset: {image_dirs}")
    print("-" * 50)
    no_image = not image_dirs or any([not image_dir for image_dir in image_dirs])
    if no_image:
        print("Image path is None, and test with random image")
        print("-" * 50)
        img_lists = [[''] * DEFAULT_COUNT] * max(1, len(image_dirs))
    else:
        dataset_name = os.path.basename(os.path.commonpath(image_dirs))
        output_dir = os.path.join(output_dir, dataset_name)
        MkdirSimple(output_dir)
        img_lists = [ReadImageList(image_dir) for image_dir in image_dirs]

    if not no_image:
        root_len = len(image_dirs[0].strip().rstrip('/'))
    print(f"Test image {MAX_COUNT} form {len(img_lists[0])} items in {len(img_lists)} group")

    for count, img_files in enumerate(zip(*img_lists), 1):
        if count > MAX_COUNT:
            break
        image, depth, depth_norm = test_image(img_files, model)
        file_name = img_files[0][root_len+1:] if not no_image else f"random_{time.time()}.jpg"
        save_image(image, depth, output_dir, file_name)

def main():
    args = GetArgs()
    output = args.output
    MkdirSimple(output)

    test_dir(args.model, [args.left_image,], output)


if __name__ == '__main__':
    main()
