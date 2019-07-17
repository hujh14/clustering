import argparse
import os
import time
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, '.')

import sklearn
from sklearn.neighbors import NearestNeighbors

import datasets
from dataset_loader import load_datasets
from html_writer import ImageHTMLWriter

def prepare_dataset(args):
    img_dir = "data/ade20k/images"
    # ann_file = "data/ade20k/annotations/instances_val.json"
    ann_file = os.path.join(args.output_dir, "predictions.json")
    emb_file = os.path.join(args.output_dir, "embeddings.npy")

    dataset = datasets.coco.COCODataset(img_dir, ann_file, transform=None)
    dataset.add_embeddings_file(emb_file)
    # dataset.filter_by_score(0.5)
    # dataset.filter_by_cat_name("person")
    return dataset


def cluster(X_train, X_test=None):
    start = time.time()
    if X_test is None:
        X_test = X_train[:1000]

    print("Clustering...", X_train.shape, X_test.shape)
    neigh = NearestNeighbors(algorithm='brute', metric='cosine').fit(X_train)
    nns = neigh.kneighbors(X_test, n_neighbors=20, return_distance=False)
    print("Done.", time.time() - start)
    return nns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./output', type=str,
                        help='output directory')
    args = parser.parse_args()

    dataset = prepare_dataset(args)

    embeddings = dataset.get_embeddings()
    nns = cluster(embeddings)

    # HTML writer
    html_fn = os.path.join(args.output_dir, "htmls/nns_all.html")
    html_writer = ImageHTMLWriter(html_fn)

    # Write to html
    for row in tqdm(nns):
        for idx in row:
            image = dataset.visualize(idx)
            html_writer.add_image(image)
        html_writer.add_line_break()


if __name__ == '__main__':
    main()
