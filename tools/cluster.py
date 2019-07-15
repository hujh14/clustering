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

def cluster(dataset, embeddings):
    neigh = NearestNeighbors(algorithm='brute', metric='cosine').fit(embeddings)
    nns = neigh.kneighbors(embeddings, n_neighbors=10, return_distance=False)
    return nns

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data = "./data"
    args.output_dir = "output/ade20k/"

    train_dataset, val_dataset = load_datasets("ade20k", args.data)

    embeddings_fn = os.path.join(args.output_dir, "embeddings.npy")
    embeddings = np.load(embeddings_fn)

    nns_fn = os.path.join(args.output_dir, "nns.npy")
    # nns = np.load(pkl_fn)
    nns = cluster(val_dataset, embeddings)
    np.save(nns_fn, nns)

    html_fn = os.path.join(args.output_dir, "nns.html")
    visualize_dataset(val_dataset, nns, html_fn)


if __name__ == '__main__':
    main()