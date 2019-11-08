"""Script to resize the full sized Cityscapes dataset to 256x128, to speed up experiments."""
import os
import argparse

from PIL import Image
from tqdm import tqdm


def save_resized_file(output_folder, file_path, imsize=[256, 128]):
    if file_path[-4:] == "json":
        return 

    img = Image.open(file_path)
    resized_image = img.resize(imsize, resample=Image.NEAREST)
    new_path = os.path.join(output_folder,
                            file_path.split('/')[-1])
    resized_image.save(new_path)


def main(root_folder, output_folder):
    # get directories of the cities
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cities = os.listdir(root_folder)
    for city in cities:
        temp_files = os.listdir(os.path.join(root_folder, city))

        output_city = os.path.join(output_folder, city)
        if not os.path.exists(output_city):
            os.makedirs(output_city)

        for file_ in tqdm(temp_files):
            save_resized_file(output_city, os.path.join(root_folder, city, file_))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('root_folder')
    parser.add_argument('output_folder')
    args = vars(parser.parse_args())

    main(args['root_folder'], args['output_folder'])
