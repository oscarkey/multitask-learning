"""Computes the mean and std of the Cityscapes input images."""
import argparse
import os

import numpy as np
from PIL import Image

_IMAGE_FILE_SUFFIX = 'leftImg8bit.png'


def _compute_stats_for_image(file_path: str) -> (np.ndarray, np.ndarray):
    with Image.open(file_path) as image:
        image_array = np.asarray(image) / 255

        mean = image_array.mean((0, 1))
        std = image_array.std((0, 1))

        assert mean.shape == (3,), f'mean had shape {mean.shape}'
        assert std.shape == (3,), f'std had shape {std.shape}'

        return mean, std


def _compute_stats_for_dir(dir_name) -> ([np.ndarray], [np.ndarray]):
    if not os.path.isdir(dir_name):
        raise ValueError(f'Directory does not exist: {dir}')

    means = []
    stds = []

    for dir_path, _, file_names in os.walk(dir_name):
        for file_name in file_names:
            if file_name.endswith(_IMAGE_FILE_SUFFIX):
                mean, std = _compute_stats_for_image(os.path.join(dir_path, file_name))
                means.append(mean)
                stds.append(std)

    return means, stds


def main(dirs: [str]):
    assert len(dirs) > 0

    means = []
    stds = []
    for dir in dirs:
        ms, ss = _compute_stats_for_dir(dir)
        means.extend(ms)
        stds.extend(ss)

    assert len(means) == len(stds)

    mean_conc = np.stack(means, axis=0)
    std_conc = np.stack(stds, axis=0)
    assert mean_conc.shape[1] == 3 and len(mean_conc.shape) == 2, f'mean_conc.shape = {mean_conc.shape}'
    assert std_conc.shape[1] == 3 and len(std_conc.shape) == 2, f'std_conc.shape = {std_conc.shape}'

    mean = mean_conc.mean(axis=0)
    std = std_conc.mean(axis=0)
    assert mean.shape == (3,), f'mean shape = {mean.shape}'
    assert std.shape == (3,), f'std shape = {std.shape}'

    print(f'Processed {len(means)} images')
    print(f'mean={mean} std={std}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', type=str, nargs='+')
    args = parser.parse_args()

    main(args.dirs)
