# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import numpy as np

import pycocotools.mask as mask_utils

min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, root, ann_file, cat_name=None, transform=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = list(sorted(self.coco.anns.keys()))
        self.transform = transform

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.area_threshold = 1000
        self.score_threshold = 0.5
        self.filter_ids()
        if cat_name:
            self.filter_category(cat_name)

    def filter_ids(self):
        # filter bad annotations
        ids = []
        for ann_id in self.ids:
            ann = self.coco.anns[ann_id]
            if not has_valid_annotation([ann]):
                continue
            if ann["area"] < self.area_threshold:
                continue
            if ann["score"] < self.score_threshold:
                continue
            ids.append(ann_id)
        self.ids = ids

        # correct image path
        for img_id in self.coco.imgs:
            img = self.coco.imgs[img_id]
            if "ade_challenge/images/" in img["file_name"]:
                img["file_name"] = img["file_name"].replace("ade_challenge/images/", "")

    def filter_category(self, cat_name):
        ids = []
        cat_name_to_cat_id = {
            cat["name"]: cat["id"] for cat in self.coco.dataset["categories"]
        }
        cat_id = cat_name_to_cat_id[cat_name]
        for ann_id in self.ids:
            ann = self.coco.anns[ann_id]
            if ann["category_id"] == cat_id:
                ids.append(ann_id)
        self.ids = ids

    def __getitem__(self, idx):
        ann_id = self.ids[idx]
        ann = self.coco.anns[ann_id]
        img = self.coco.imgs[ann["image_id"]]

        input = self.prepare_input(img, ann)
        target = self.prepare_target(ann)
        return input, target, idx

    def __len__(self):
        return len(self.ids)

    def prepare_input(self, img, ann):
        image_path = os.path.join(self.root, img["file_name"])
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        bbox = ann["bbox"]
        image = crop_bbox(image, bbox, margin=0.2)
       
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
        return image

    def prepare_input_with_mask(self, img, ann):
        image_path = os.path.join(self.root, img["file_name"])
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        mask = mask_utils.decode(ann["segmentation"])  # [h, w, n]
        bbox = ann["bbox"]

        image = crop_bbox(image, bbox, margin=0.2)
        mask = crop_bbox(mask, bbox, margin=0.2)
        mask = mask * 255

        if self.transform:
            image = Image.fromarray(image)
            mask = Image.fromarray(mask)
            # Hack to ensure transform is the same
            seed = random.randint(0,2**32)
            random.seed(seed)
            image = self.transform(image)
            random.seed(seed)
            mask = self.transform(mask)
            image = torch.cat([image, mask])
        return image

    def prepare_target(self, ann):
        target = self.json_category_id_to_contiguous_id[ann["category_id"]]
        return target

    def prepare_all_targets(self):
        targets = []
        for ann_id in self.ids:
            ann = self.coco.anns[ann_id]
            target = self.prepare_target(ann)
            targets.append(target)
        return targets

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

def crop_bbox(image, bbox, margin=0.2):
    x, y, w, h = bbox
    # Add margin
    space = margin * (w + h) / 2
    x = round(x - space)
    y = round(y - space)
    w = round(w + space * 2)
    h = round(h + space * 2)

    x0 = min(max(0, x  ), image.shape[1])
    x1 = min(max(0, x+w), image.shape[1])
    y0 = min(max(0, y  ), image.shape[0])
    y1 = min(max(0, y+h), image.shape[0])
    crop = image[y0:y1, x0:x1]

    # Pad with zeros
    pad = np.zeros((h, w, 3), dtype='uint8')
    if image.ndim == 2:
        pad = np.zeros((h, w), dtype='uint8')
    pad_l = max(-x, 0)
    pad_t = max(-y, 0)
    pad[pad_t:pad_t + crop.shape[0], pad_l:pad_l + crop.shape[1]] = crop
    return pad

if __name__ == '__main__':
    data_dir = "./data"
    root = os.path.join(data_dir, "ade20k/images")
    val_ann_file = os.path.join(data_dir, "ade20k/annotations/predictions_val.json")

    dataset = COCODataset(
        root, val_ann_file, 
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.RandomResizedCrop(224, scale=(0.5,1.)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    print("Dataset size:", len(dataset))

    for inp, target, idx in dataset:
        print("Input shape:", inp.shape)
        print("Target:", target)
        print("Index:", idx)
        print("Img info", dataset.get_img_info(idx))
        
        image = np.array(inp.numpy() * 255, dtype="uint8")
        image = np.transpose(image, (1,2,0))

        if image.shape[2] == 4:
            # Visualize mask channel
            image_c = np.array(image[:,:,:3], dtype="float")
            iszero = np.nonzero(image[:,:,3] == 0)
            image_c[iszero[0], iszero[1], :] *= 0.3
            image = np.array(image_c, dtype="uint8")

        image = Image.fromarray(image)
        image.show()
        input("Press Enter to continue...")
