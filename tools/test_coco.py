import argparse
import os
import random
import shutil
import time
import warnings
import sys
sys.path.insert(0, '.')

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

from PIL import Image
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data = "./data"
    args.workers = 1
    args.batch_size = 128

    # train_img_dir = "coco/train2017"
    # train_ann_file = "coco/annotations/instances_train2017.json"
    # val_img_dir = "coco/val2017"
    # val_ann_file = "coco/annotations/instances_val2017.json"
    train_img_dir = "ade20k/images"
    train_ann_file = "ade20k/annotations/instances_train.json"
    val_img_dir = "ade20k/images"
    val_ann_file = "ade20k/annotations/instances_val.json"

    train_img_dir = os.path.join(args.data, train_img_dir)
    train_ann_file = os.path.join(args.data, train_ann_file)
    val_img_dir = os.path.join(args.data, val_img_dir)
    val_ann_file = os.path.join(args.data, val_ann_file)

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.coco.COCODataset(train_img_dir, train_ann_file, transform=train_transforms)
    val_dataset = datasets.coco.COCODataset(val_img_dir, val_ann_file, transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    data_loader = val_loader

    for i, (images, target, index) in enumerate(data_loader):
        print(images.shape, target.shape, index.shape)
        for image, targ, idx in zip(images, target, index):
            print(image.shape, targ, idx)
            print(data_loader.dataset.get_info(idx))
            image = np.array(image.numpy() * 255, dtype="uint8")
            image = np.transpose(image, (1,2,0))
            image = Image.fromarray(image)
            image.show()
            input("Press Enter to continue...")
            break

if __name__ == '__main__':
    main()