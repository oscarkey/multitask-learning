import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def check_file(file_path, imsize=[256, 128]):
    if file_path[-4:] == "json":
        return

    image_array = np.asarray(Image.open(file_path), dtype=np.float32)
    # Rescale the image from [0,255] to [0,1].
    image_array = image_array / 255 * 2 - 1

    if 'left' in file_path:
        assert len(image_array.shape) == 3, f'image_array should have 2 dimensions, the file path is {file_path} and the shape is {image_array.shape}'

    elif 'instance' in file_path:
        assert len(image_array.shape) == 2, f'image_array should have 2 dimensions, the file path is {file_path} and the shape is {image_array.shape}'

    elif 'labelIds' in file_path:
        assert len(image_array.shape) == 2, f'image_array should have 2 dimensions, the file path is {file_path} and the shape is {image_array.shape}'

    else:
        return




def main(root_folder):
    # get directories of the cities

    cities = os.listdir(root_folder)
    for city in cities:
        temp_files = os.listdir(os.path.join(root_folder, city))

        for file_ in tqdm(temp_files):
            check_file(os.path.join(root_folder, city, file_))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('root_folder')
    #parser.add_argument('-height', type=int)
    #parser.add_argument('-width', type=int)
    args = vars(parser.parse_args())

    main(args['root_folder'])
