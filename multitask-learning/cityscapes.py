"""Loads the Cityscapes dataset for use with Pytorch."""

import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CityscapesDataset(Dataset):
    """A Dataset which loads the Cityscapes dataset from disk.

    The data files should be named as follows: {city}_{id}_{frame}_{type}.{ext}
    where each image exists for at least the instanceIds and labelIds types.
    This is conveniently the naming scheme of the data available from the Cityscapes website.

    As Cityscapes already splits the data into train/val/test, you may want to create an instance of
    this class for each (though maybe we want different splits...).
    """

    def __init__(self, root_dir: str):
        # TODO: Potentially add transform for resizing data.
        self._root_dir = root_dir
        self._file_prefixes = self._find_file_prefixes(root_dir)

    @staticmethod
    def _find_file_prefixes(root_dir: str) -> [str]:
        """Finds data files under the given path and returns the prefix of the path to each.

        Walks the directory tree looking for data files. Several files exist for each image
        (segmentation, instance ids, etc.) which all share the same prefix. This function
        returns the prefix only once.

        :return the {path under root}/{city}_{id}_{frame} portion of the path to each file
        """
        # As several files exist per prefix, use a set to deduplicate them.
        file_prefixes = set()

        for (path, dirs, files) in os.walk(root_dir):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext == '.png':
                    file_prefixes.add(CityscapesDataset._get_file_prefix(path, file))

        return list(file_prefixes)

    @staticmethod
    def _get_file_prefix(directory: str, file_name: str) -> str:
        # The format is {city}_{seq:0>6}_{frame:0>6}_{type1}_{type2}.{ext}
        parts = file_name.split('_')
        assert len(parts) == 5 or len(parts) == 4, 'File name not as expected: ' + str(parts)
        prefix = parts[0] + '_' + parts[1] + '_' + parts[2]
        return os.path.join(directory, prefix)

    def __getitem__(self, index: int):
        image_file = self._get_file_path_for_index(index, 'leftImg8bit')
        label_file = self._get_file_path_for_index(index, 'labelIds')

        image_array = np.asarray(Image.open(image_file), dtype=np.float32)
        # Rescale the image from [0,255] to [0,1].
        image_array = image_array / 255 * 2 - 1
        assert len(image_array.shape) == 3, 'image_array should have 3 dimensions'+ image_file

        label_array = np.asarray(Image.open(label_file), dtype=np.long)
        assert len(label_array.shape) == 2, 'label_array should have 2 dimensions'+ label_file

        instance_file = self._get_file_path_for_index(index, 'instanceIds')
        instance_image = Image.open(instance_file)
        instance_vecs, instance_mask = self._compute_centroid_vectors(np.asarray(instance_image))

        # We load the images as H x W x channel, but we need channel x H x W.
        axis_order = (2, 0, 1)

        return (np.transpose(image_array, axis_order),
                label_array,
                np.transpose(instance_vecs, axis_order),
                instance_mask)

    def _get_file_path_for_index(self, index: int, type: str) -> str:
        path_prefix = self._file_prefixes[index]
        files = glob.glob(f'{path_prefix}*_{type}.png')
        assert len(files) > 0, 'Expect at least one file for the given type.'
        assert len(files) == 1, 'Only expect one file for the given type.'
        return files[0]

    @staticmethod
    def _compute_centroid_vectors(instance_image):
        """For each pixel, calculate the vector from that pixel to the centre of its instance.

        :return a pair of a matrix containing the distance vector to every pixel, and a mask
        identifying which pixels are associated with an instance
        """
        # Each pixel in the image is of one of two formats:
        # 1) If the pixel does not belong to an instance:
        #    The id of the class the pixel belongs to
        # 2) If the pixel does belong to an instance:
        #    id x 1000 + instance id

        # For each instance, find all pixels associated with it and compute the centre.
        # Add an extra dimension for each pixel containing the coordinates of the associated centre.
        centroids = np.zeros(instance_image.shape + (2,))
        for value in np.unique(instance_image):
            xs, ys = np.where(instance_image == value)
            centroids[xs, ys] = np.array((np.floor(np.mean(xs)), np.floor(np.mean(ys))))

        # Calculate the distance from the x,y coordinates of the pixel to the coordinates of the
        # centre of its associated instance.
        vecs = np.zeros(instance_image.shape + (2,))
        g1, g2 = np.meshgrid(range(instance_image.shape[1]), range(instance_image.shape[0]))
        vecs[:, :, 0] = g1
        vecs[:, :, 1] = g2
        vecs = vecs - centroids

        mask = np.ma.masked_where(instance_image >= 1000, instance_image)
        mask = np.asarray(mask.mask, dtype=np.uint8)
        mask = np.stack((mask, mask))
        return vecs, mask

    def __len__(self):
        return len(self._file_prefixes)


def get_loader_from_dir(root_dir: str, config):
    """Creates a DataLoader for Cityscapes from the given root directory.

    Will load any data file in any sub directory under the root directory.
    """

    return get_loader(CityscapesDataset(root_dir), config)


def get_loader(dataset: Dataset, config):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False)


if __name__ == '__main__':
    root = '/Users/oscar/Downloads/gtFine_trainvaltest/gtFine/train/'
    test = CityscapesDataset(root)
    print(test[1])
