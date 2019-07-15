import argparse
import os
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, '.')

import torch
import torchvision.transforms as transforms

import datasets

def load_train_dataset(img_dir, ann_file):
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.coco.COCODataset(img_dir, ann_file, transform=train_transforms)
    return train_dataset

def load_val_dataset(img_dir, ann_file):
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    val_dataset = datasets.coco.COCODataset(img_dir, ann_file, transform=val_transforms)
    return val_dataset

def load_datasets(dataset_name, data_dir):
    if dataset_name == "coco":
        train_img_dir = "coco/train2017"
        train_ann_file = "coco/annotations/instances_train2017.json"
        val_img_dir = "coco/val2017"
        val_ann_file = "coco/annotations/instances_val2017.json"
    elif dataset_name == "ade20k":
        train_img_dir = "ade20k/images"
        train_ann_file = "ade20k/annotations/instances_train.json"
        val_img_dir = "ade20k/images"
        val_ann_file = "ade20k/annotations/instances_val.json"
    else:
        raise Exception("Dataset name not recognized")

    train_img_dir = os.path.join(data_dir, train_img_dir)
    train_ann_file = os.path.join(data_dir, train_ann_file)
    val_img_dir = os.path.join(data_dir, val_img_dir)
    val_ann_file = os.path.join(data_dir, val_ann_file)

    train_dataset = load_train_dataset(train_img_dir, train_ann_file)
    val_dataset = load_train_dataset(val_img_dir, val_ann_file)
    return train_dataset, val_dataset



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data = "./data"
    args.workers = 1
    args.batch_size = 128
    
    train_dataset, val_dataset = load_datasets("ade20k", args.data)

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



