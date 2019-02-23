import os
from tqdm import tqdm
from scipy.misc import imread, imsave, imresize
import argparse


def save_resized_file(output_folder, file_path, imsize=[128, 256]):
    resized_image = imresize(imread(file_path), imsize)
    new_path = os.path.join(output_folder,
                            file_path.split('/')[-1])
    imsave(new_path, resized_image)


def main(root_folder, output_folder):
    # get directories of the cities
    cities = os.listdir(root_folder)
    for city in cities:
        temp_files = os.listdir(os.path.join(root_folder, city))
        for file_ in tqdm(temp_files):
            save_resized_file(output_folder, os.path.join(root_folder, city, file_))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-output_folder', type=str)
    #parser.add_argument('-height', type=int)
    #parser.add_argument('-width', type=int)
    parser.add_argument('-root_folder', type=str)
    args = vars(parser.parse_args())
    print(args)
    main(args['root_folder'], args['output_folder'])
