import argparse
import os
import uuid
import numpy as np
import sys
sys.path.insert(0, '.')

from dataset_loader import load_dataset

class ImageHTMLWriter:

    def __init__(self, html_fn, img_height=200):
        self.html_fn = html_fn
        self.output_dir = os.path.dirname(html_fn)
        self.img_dir = os.path.join(self.output_dir, "html_images")

        self.img_height = img_height

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
        img_fn = os.path.join(self.img_dir, "{}.jpg".format(uuid.uuid4().hex))
        image.save(img_fn)

        path = os.path.relpath(img_fn, os.path.dirname(self.html_fn))
        with open(self.html_fn, "a") as f:
            tag = "<img src=\"{}\" height=\"{}\">".format(path, self.img_height)
            f.write(tag + "\n")

    def add_line_break(self):
        with open(self.html_fn, "a") as f:
            tag = "<br>"
            f.write(tag + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.output_dir = "output/ade20k/"

    html_fn = os.path.join(args.output_dir, "htmls/test_vis.html")
    html_writer = ImageHTMLWriter(html_fn)

    dataset_name = "ade20k_val"
    dataset = load_dataset(dataset_name, training=True)
    idxs = range(20)
    for idx in range(20):
        image = dataset.visualize(idx)
        html_writer.add_image(image)


