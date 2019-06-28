import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import datasets
import torchvision.models as models

def main():
    data_dir = "./data"
    root = os.path.join(data_dir, "ade20k/images")
    train_ann_file = os.path.join(data_dir, "ade20k/annotations/instances_train.json")
    val_ann_file = os.path.join(data_dir, "ade20k/annotations/instances_val.json")

    train_dataset = datasets.coco.COCODataset(
        val_ann_file, root,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=False,
        num_workers=1, pin_memory=True, sampler=None)

    for i, (input, target) in enumerate(train_loader):
    	a = input[0]
    	b = target[0]
    	print(a.shape)
    	print(b)

if __name__ == '__main__':
    main()