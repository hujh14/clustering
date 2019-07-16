import argparse
import os
import time
import numpy as np
import sys
sys.path.insert(0, '.')

import sklearn
from sklearn.neighbors import NearestNeighbors

import datasets
from dataset_loader import load_datasets
from dataset_visualizer import visualize_dataset

def cluster(embeddings):
    start = time.time()
    print("Clustering embeddings", embeddings.shape)
    neigh = NearestNeighbors(algorithm='brute', metric='cosine').fit(embeddings)
    nns = neigh.kneighbors(embeddings, n_neighbors=10, return_distance=False)
    print("Done.", time.time() - start)
    return nns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', default="ade20k", type=str,
                        help='Dataset name')
    args = parser.parse_args()
    args.output_dir = os.path.join("output", args.dataset_name)

    img_dir = "data/coco/val2017"
    ann_file = os.path.join(args.output_dir, "predictions.json")
    embeddings_file = os.path.join(args.output_dir, "embeddings.npy")

    dataset = datasets.coco.COCODataset(img_dir, ann_file, transform=None)
    dataset.add_embeddings_file(embeddings_file)
    dataset.filter_by_score(0.5)
    dataset.filter_by_cat_name("person")
    embeddings = dataset.get_embeddings()

    nns = cluster(embeddings)
    np.random.shuffle(nns)

    # Visualize to html
    html_fn = os.path.join(args.output_dir, "nns.html")
    visualize_dataset(dataset, nns, html_fn)


if __name__ == '__main__':
    main()
