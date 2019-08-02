import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter, ImageChops
import os
import random
import numpy as np

from pycocotools.coco import COCO
import pycocotools.mask as mask_utils

class PlacesDataset(torch.utils.data.Dataset):

    def __init__(self, root, ann_dir, transform=None):
        self.coco = build_coco(ann_dir)
        self.root = root
        self.ann_dir = ann_dir
        self.transform = transform

        self.ids = list(sorted(self.coco.anns.keys()))
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        inp = self.prepare_input(idx)
        target = self.prepare_target(idx)
        return inp, target, idx

    def get_image_vis(self, idx):
        image = self.get_image(idx)
        mask = self.get_mask(idx)
        mask = Image.fromarray(mask * 255)

        erosion_mask = mask.filter(ImageFilter.MinFilter(3))
        edges = ImageChops.difference(mask, erosion_mask)

        enhancer = ImageEnhance.Brightness(image)
        image_vis = enhancer.enhance(0.5)
        image_vis.paste(image, mask=mask)
        image_vis.paste(edges, mask=edges)
        return image_vis

    def get_image(self, idx):
        img = self.get_img_info(idx)
        image_path = os.path.join(self.root, img["file_name"])
        image = Image.open(image_path).convert('RGB')
        return image

    def get_mask(self, idx):
        ann = self.get_ann_info(idx)
        mask = self.coco.annToMask(ann)
        return mask

    def get_bbox(self, idx):
        ann = self.get_ann_info(idx)
        bbox = ann["bbox"]
        return bbox

    def get_img_info(self, idx):
        ann_id = self.ids[idx]
        ann = self.coco.anns[ann_id]
        img = self.coco.imgs[ann["image_id"]]
        return img

    def get_ann_info(self, idx):
        ann_id = self.ids[idx]
        ann = self.coco.anns[ann_id]
        return ann

    def get_cat_info(self, idx):
        ann_id = self.ids[idx]
        ann = self.coco.anns[ann_id]
        cat = self.coco.cats[ann["category_id"]]
        return cat


def build_coco(ann_dir):
    images = []
    annotations = []
    categories = []

    split_dirs = sorted(os.listdir(ann_dir))
    for n, split_dir in enumerate(split_dirs[:1]):
        ann_file = os.path.join(ann_dir, split_dir, "predictions.json")
        with open(ann_file) as f:
            data = json.load(f)
            images.append(data["images"])
            annotations.append(data["annotations"])
            if n == 0:
                categories = data["categories"]

    coco = COCO()
    coco.dataset["images"] = images
    coco.dataset["annotations"] = annotations
    coco.dataset["categories"] = categories
    coco.createIndex()
    return coco

