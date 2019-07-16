import argparse
import os
import numpy as np
import sys
sys.path.insert(0, '.')

import sklearn
from sklearn.neighbors import NearestNeighbors

import datasets
from dataset_loader import load_datasets
from dataset_visualizer import visualize_dataset

def cluster(embeddings):
    neigh = NearestNeighbors(algorithm='brute', metric='cosine').fit(embeddings)
    nns = neigh.kneighbors(embeddings, n_neighbors=10, return_distance=False)
    return nns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', default="ade20k", type=str,
                        help='Dataset name')
    args = parser.parse_args()
    args.output_dir = os.path.join("output", args.dataset_name)

    embeddings_fn = os.path.join(args.output_dir, "embeddings.npy")
    embeddings = np.load(embeddings_fn)

    nns_fn = os.path.join(args.output_dir, "nns.npy")
    nns = cluster(val_dataset, embeddings)
    np.save(nns_fn, nns)
    # nns = np.load(pkl_fn)

    # Visualize to html
    html_fn = os.path.join(args.output_dir, "nns.html")
    train_dataset, val_dataset = load_datasets(args.dataset_name)
    visualize_dataset(val_dataset, nns, html_fn)


if __name__ == '__main__':
    main()