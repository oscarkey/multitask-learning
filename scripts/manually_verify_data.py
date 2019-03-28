"""Visualizes random sample of training data to allow human to verify it."""
import argparse
import os
import random

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def check_file(path: str, image_file_name: str):
    label_file_name = image_file_name.replace('leftImg8bit', 'gtFine_labelIds')
    image = Image.open(os.path.join(path, image_file_name))
    labels = np.asarray(Image.open(os.path.join(path, label_file_name)))

    _, axes = plt.subplots(1, 2)

    axes[0].imshow(image)
    axes[1].imshow(np.where(labels != 255, labels, np.zeros_like(labels)))

    plt.savefig(f'test_{image_file_name}')


def main(directory: str):
    checked = 0
    total = 0
    for path, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith('leftImg8bit.png'):
                total += 1

                if random.random() < 0.1:
                    check_file(path, file_name)
                    checked += 1

    print(f'Checked {checked} / {total} images.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, nargs='?')
    args = parser.parse_args()

    main(args.directory)
