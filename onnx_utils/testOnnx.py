import onnxruntime
import torch
from onnxmodel import ONNXModel
import cv2
from dataset.utils import np2torch, np2float
import numpy as np
from colormap import apply_colormap
import torchvision

model = ONNXModel("hitnet_sf_finalpass_640_400.onnx")

lp = "/data/ljdong/Depth_Estimate/image/left/0000000000.png"
rp = "/data/ljdong/Depth_Estimate/image/right/0000000000.png"
left = cv2.imread(str(lp), cv2.IMREAD_COLOR)
right = cv2.imread(str(rp), cv2.IMREAD_COLOR)

left = np2float(left, True)
right = np2float(right, True)

left = np.expand_dims(left, axis=0).astype(np.float32)
right = np.expand_dims(right, axis=0).astype(np.float32)

left = left * 2 - 1
right = right * 2 - 1

print(left.shape)
# output = model.forward({"L":left, "R":right})
output = model.forward2((left, right))

disp = output[0][:, 0:1]
disp = torch.from_numpy(disp)
disp = np.clip(disp / 192 * 255, 0, 255).long()

disp = apply_colormap(disp)
output1 = [torch.from_numpy(left), disp]
print(left.shape, disp.shape)
output1 = torch.cat(output1, dim=0)
torchvision.utils.save_image(output1, "disp_from_onnx.png", nrow=1)