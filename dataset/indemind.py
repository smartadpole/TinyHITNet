import cv2
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset

from dataset.utils import *


class INDEMINDataset(Dataset):
    def __init__(
        self,
        image_list,
        root,
        crop_size=None,
        training=False,
        augmentation=False,
    ):
        super().__init__()
        self.root = Path(root)
        with open(self.root / "train_cam0.txt", "rt") as fp:
            self.file_list_left = sorted([Path(line.strip()) for line in fp])
        with open(self.root / "train_cam1.txt", "rt") as fp:
            self.file_list_right = sorted([Path(line.strip()) for line in fp])
        with open(self.root / "disp.txt", "rt") as fp:
            self.file_list_disp = sorted([Path(line.strip()) for line in fp])
        with open(self.root / "slant.txt", "rt") as fp:
            self.file_list_slant = sorted([Path(line.strip()) for line in fp])
        self.crop_size = crop_size
        self.training = training
        self.augmentation = augmentation

    def __len__(self):
        return len(self.file_list_left)

    def __getitem__(self, index):
        left_path = self.file_list_left[index]
        right_path = self.file_list_right[index]
        disp_path = self.file_list_disp[index]
        dxy_path = self.file_list_slant[index]

        data = {
            "left": np2torch(cv2.imread(str(left_path), cv2.IMREAD_COLOR), bgr=True),
            "right": np2torch(cv2.imread(str(right_path), cv2.IMREAD_COLOR), bgr=True),
            "disp": np2torch(
                cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
                / 256
            ),
            "dxy": np2torch(np.load(dxy_path), t=False),
        }
        if self.crop_size is not None:
            data = crop_and_pad(data, self.crop_size, self.training)
        if self.training and self.augmentation:
            data = augmentation(data, self.training)
        return data


if __name__ == "__main__":
    import torchvision
    from colormap import apply_colormap, dxy_colormap

    dataset = INDEMINDataset(
        "lists/kitti2015_train.list",
        "/home/tiger/KITTIStereo/KITTI2015/training",
        training=True,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for ids, data in enumerate(loader):
        disp = data["disp"]
        disp = torch.clip(disp / 192 * 255, 0, 255).long()
        disp = apply_colormap(disp)

        dxy = data["dxy"]
        dxy = dxy_colormap(dxy)
        output = torch.cat((data["left"], data["right"], disp, dxy), dim=0)
        torchvision.utils.save_image(output, "{:06d}.png".format(ids), nrow=1)
