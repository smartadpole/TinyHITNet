#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: onnxmodel.py
@time: 2021/2/2 下午5:55
@desc:
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import argparse
import numpy as np

# -*-coding: utf-8 -*-

import os, sys
sys.path.append("/work/LIB/CPP/onnx/libonnxruntime.so")
import onnxruntime


class ONNXModel():
    def __init__(self, onnx_file):
        self.onnx_session = onnxruntime.InferenceSession(onnx_file, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                                                               'CPUExecutionProvider'])

        print(f"ONNX Runtime version: {onnxruntime.__version__}")
        print(f"Device: {onnxruntime.get_device()}")
        print(f"Available providers: {self.onnx_session.get_providers()}")

        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        self.print_IO()

    def print_IO(self,):
        # 查看模型的输入信息
        print("Model Inputs:")
        for input in self.onnx_session.get_inputs():
            print(f"Name: {input.name}")
            print(f"Shape: {input.shape}")
            print(f"Data Type: {input.type}")
            print("-" * 50)
            break

        # 查看模型的输出信息
        print("Model Outputs:")
        for output in self.onnx_session.get_outputs():
            print(f"Name: {output.name}")
            print(f"Shape: {output.shape}")
            print(f"Data Type: {output.type}")
            print("-" * 50)
            break

    def get_output_name(self, onnx_session):
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image:np.ndarray):
        input_feed = {}
        for name in input_name:
            input_feed[name] = image
        return input_feed

    def forward(self, image:np.ndarray):
        input_feed = self.get_input_feed(self.input_name, image)
        scores = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return scores



def to_numpy(tensor):
    print(tensor.device)
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()