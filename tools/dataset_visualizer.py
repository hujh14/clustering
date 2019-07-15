import argparse
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import sys
sys.path.insert(0, '.')

from dataset_loader import load_datasets

class ImageHTMLWriter:

    def __init__(self, html_fn, img_height=300):
        self.html_fn = html_fn
        self.output_dir = os.path.dirname(html_fn)
        self.img_dir = os.path.join(self.output_dir, "html_images")

        self.img_height = img_height
        self.img_paths = []

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        if os.path.exists(self.html_fn):
            os.remove(self.html_fn)

        self.init_html()

    def init_html(self):
        with open(self.html_fn, "a") as f:
            tag = "<body style=\"white-space: nowrap;\">"
            f.write(tag + "\n")

    def add_image(self, image):
        img_fn = os.path.join(self.img_dir, "{}.jpg".format(len(self.img_paths)))
        image.save(img_fn)
        self.img_paths.append(img_fn)

        path = os.path.relpath(img_fn, os.path.dirname(self.html_fn))
        with open(self.html_fn, "a") as f:
            tag = "<img src=\"{}\" height=\"{}\">".format(path, self.img_height)
            f.write(tag + "\n")

    def line_break(self):
        with open(self.html_fn, "a") as f:
            tag = "<br>"
            f.write(tag + "\n")


def visualize_dataset(dataset, idxs, html_fn):
    html_writer = ImageHTMLWriter(html_fn, img_height=200)

    grid_idxs = np.array(idxs, dtype=int)
    if np.ndim(grid_idxs) == 1:
        grid_idxs = grid_idxs[np.newaxis, :]

    for rows_idxs in tqdm(grid_idxs):
        for idx in rows_idxs:
            image = dataset.get_image(idx)
            mask = dataset.get_mask(idx)
            mask = Image.fromarray(mask * 255)

            enhancer = ImageEnhance.Brightness(image)
            image_vis = enhancer.enhance(0.2)
            image_vis.paste(image, mask=mask)

            html_writer.add_image(image_vis)
        html_writer.line_break()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data = "./data"
    args.output_dir = "output/ade20k/"

    train_dataset, val_dataset = load_datasets("ade20k", args.data)

    out_fn = os.path.join(args.output_dir, "test.html")
    vis_dataset(val_dataset, [1,2,3,4], out_fn)


