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
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    def __init__(self, root, ann_file, transform=None):
        super(COCODataset, self).__init__(root, ann_file)
        self.ids = list(sorted(self.coco.anns.keys()))
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.transform = transform

    def __getitem__(self, idx):
        inp = self.prepare_input(idx)
        target = self.prepare_target(idx)
        return inp, target, idx

    def __len__(self):
        return len(self.ids)

    def filter_by_score(self, score):
        ids = []
        for ann_id in self.ids:
            ann = self.coco.anns[ann_id]
            if ann["score"] >= score:
                ids.append(ann_id)
        self.ids = ids

    def filter_by_cat_name(self, cat_name):
        ids = []
        for ann_id in self.ids:
            ann = self.coco.anns[ann_id]
            cat = self.coco.cats[ann["category_id"]]
            if cat["name"] == cat_name:
                ids.append(ann_id)
        self.ids = ids

    def add_embeddings_file(self, embeddings_fn):
        self.embeddings = np.load(embeddings_fn)
        assert len(self.embeddings) == len(self.coco.anns.keys())

    def get_embeddings(self):
        embeddings = []
        for ann_id in self.ids:
            embeddings.append(self.embeddings[ann_id-1])
        return np.array(embeddings)

    def prepare_input(self, idx):
        image = self.get_image(idx)
        mask = self.get_mask(idx)
        bbox = self.get_bbox(idx)

        image = np.array(image)
        mask = mask * 255
        image = crop_bbox(image, bbox, margin=0.2)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        return image

    def prepare_target(self, idx):
        ann = self.get_ann_info(idx)
        target = self.json_category_id_to_contiguous_id[ann["category_id"]]
        return target

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

def crop_bbox(image, bbox, margin=0.2):
    x, y, w, h = bbox
    # Add margin
    space = margin * (w + h) / 2
    x = int(round(x - space))
    y = int(round(y - space))
    w = int(round(w + space * 2))
    h = int(round(h + space * 2))

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
