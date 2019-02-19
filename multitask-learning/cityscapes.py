"""Loads the Cityscapes dataset for use with Pytorch."""

import glob
import os

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
                file_prefixes.add(CityscapesDataset._get_file_prefix(path, file))

        return list(file_prefixes)

    @staticmethod
    def _get_file_prefix(directory: str, file_name: str) -> str:
        # The format is {city}_{seq:0>6}_{frame:0>6}_{type1}_{type2}.{ext}
        parts = file_name.split('_')
        assert len(parts) == 5, 'File name not as expected: ' + str(parts)
        prefix = parts[0] + '_' + parts[1] + '_' + parts[2]
        return os.path.join(directory, prefix)

    def __getitem__(self, index: int):
        label_file = self._get_file_path_for_index(index, 'labelIds')
        instance_file = self._get_file_path_for_index(index, 'instanceIds')

        # TODO: Process files into tensors.
        raise NotImplementedError()

    def _get_file_path_for_index(self, index: int, type: str) -> str:
        path_prefix = os.path.join(self._root_dir, self._file_prefixes[index])
        files = glob.glob(f'{path_prefix}*_{type}.png')
        assert len(files) == 1, 'Only expect one file for the given type.'
        return files[0]

    def __len__(self):
        return len(self._file_prefixes)


if __name__ == '__main__':
    root = '/Users/oscar/Downloads/gtFine_trainvaltest/gtFine/train/'
    test = CityscapesDataset(root)
