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
        "places": {
            "img_dir": "/data/vision/torralba/ade20k-places/data",
            "ann_file": "/data/vision/oliva/scenedataset/scaleplaces/active_projects/maskrcnn_embedding/output/coco_places/inference"
        },
    }

def load_dataset(dataset_name, training=False):
    if "coco" in dataset_name or "ade20k" in dataset_name:
        dataset = load_coco_dataset(dataset_name, training)
    elif "places" in dataset_name:
        dataset = load_places_dataset(dataset_name, training)
    else:
        dataset = None

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    if training:
        dataset.transform = train_transforms
    else:
        dataset.transform = val_transforms
    return dataset

def load_places_dataset(dataset_name, training=False):
    img_dir = DATASETS[dataset_name]["img_dir"]
    ann_file = DATASETS[dataset_name]["ann_file"]
    return datasets.places.PlacesDataset(img_dir, ann_file)

def load_coco_dataset(dataset_name, training=False):
    img_dir = DATASETS[dataset_name]["img_dir"]
    ann_file = DATASETS[dataset_name]["ann_file"]

    # Assume in data directory
    img_dir = os.path.join(DATA_DIR, img_dir)
    ann_file = os.path.join(DATA_DIR, ann_file)
    return datasets.coco.COCODataset(img_dir, ann_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.workers = 1
    args.batch_size = 128
    
    dataset_name = "places"
    dataset = load_dataset(dataset_name, training=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

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



