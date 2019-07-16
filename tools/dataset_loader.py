import argparse
import os
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, '.')

import torch
import torchvision.transforms as transforms

import datasets

DATA_DIR = "data"
DATASETS = {
        "coco_2017_train": {
            "img_dir": "coco/train2017",
            "ann_file": "coco/annotations/instances_train2017.json"
        },
        "coco_2017_val": {
            "img_dir": "coco/val2017",
            "ann_file": "coco/annotations/instances_val2017.json"
        },
        "ade20k_train": {
            "img_dir": "ade20k/images",
            "ann_file": "ade20k/annotations/instances_train.json"
        },
        "ade20k_val": {
            "img_dir": "ade20k/images",
            "ann_file": "ade20k/annotations/instances_val.json"
        },
    }

def load_train_dataset(dataset_name):
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    img_dir = os.path.join(DATA_DIR, DATASETS[dataset_name]["img_dir"])
    ann_file = os.path.join(DATA_DIR, DATASETS[dataset_name]["ann_file"])
    train_dataset = datasets.coco.COCODataset(img_dir, ann_file, transform=train_transforms)
    return train_dataset

def load_val_dataset(dataset_name):
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_dir = os.path.join(DATA_DIR, DATASETS[dataset_name]["img_dir"])
    ann_file = os.path.join(DATA_DIR, DATASETS[dataset_name]["ann_file"])
    val_dataset = datasets.coco.COCODataset(img_dir, ann_file, transform=val_transforms)
    return val_dataset

def load_datasets(dataset_name):
    if "coco" in dataset_name:
        train_dataset_name = "coco_2017_train"
        val_dataset_name = "coco_2017_val"
    elif "ade20k" in dataset_name:
        train_dataset_name = "ade20k_train"
        val_dataset_name = "ade20k_val"
    else:
        raise Exception("Dataset name not recognized: "+ dataset_name)

    train_dataset = load_train_dataset(train_dataset_name)
    val_dataset = load_train_dataset(val_dataset_name)
    return train_dataset, val_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.workers = 1
    args.batch_size = 128
    
    train_dataset, val_dataset = load_datasets("ade20k")

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



