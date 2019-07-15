import argparse
import os
import numpy as np
import pickle as pkl
import sys
sys.path.insert(0, '.')

import sklearn
from sklearn.neighbors import NearestNeighbors

import datasets

def save(obj, obj_fn):
    with open(obj_fn, 'wb') as f:
        pkl.dump(obj, f)

def load(obj_fn):
    with open(obj_fn, 'rb') as f:
        return pkl.load(f)

def cluster(dataset, embeddings, output_dir):
    # neigh = NearestNeighbors(algorithm='brute', metric='cosine').fit(embeddings)
    # kns = neigh.kneighbors(embeddings, n_neighbors=10, return_distance=False)

    tmp_fn = os.path.join(output_dir, "nns.pkl")
    # save(kns, tmp_fn)
    kns = load(tmp_fn)

    for ns in kns:
        cats = []
        for idx in ns:
            img, ann, cat = dataset.get_info(idx)
            cats.append(cat["name"])
        print(cats)
    


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data = "./data"
    args.output_dir = "output/ade20k/"

    embeddings_fn = os.path.join(args.output_dir, "embeddings.npy")
    embeddings = np.load(embeddings_fn)

    # Data loading code
    train_img_dir = "ade20k/images"
    train_ann_file = "ade20k/annotations/instances_train.json"
    val_img_dir = "ade20k/images"
    val_ann_file = "ade20k/annotations/instances_val.json"

    train_img_dir = os.path.join(args.data, train_img_dir)
    train_ann_file = os.path.join(args.data, train_ann_file)
    val_img_dir = os.path.join(args.data, val_img_dir)
    val_ann_file = os.path.join(args.data, val_ann_file)

    # train_dataset = datasets.coco.COCODataset(train_img_dir, train_ann_file, transform=None)
    val_dataset = datasets.coco.COCODataset(val_img_dir, val_ann_file, transform=None)

    cluster(val_dataset, embeddings, args.output_dir)

if __name__ == '__main__':
    main()