import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

import cityscapes


def save_mask_instances_file(output_folder, file_path):
    instance_array = np.asarray(Image.open(file_path), dtype=np.float32)
    file_path = file_path.replace('instanceIds', 'instanceMask')
    instance_vecs, instance_mask = cityscapes.compute_centroid_vectors(instance_array)
    new_path = os.path.join(output_folder, file_path.split('/')[-1])
    instance = {"mask": instance_mask, "vec": instance_vecs}
    np.save(new_path, instance)


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
            if "instanceIds" in file_:
                save_mask_instances_file(output_city, os.path.join(root_folder, city, file_))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_folder')
    # parser.add_argument('output_folder')
    # parser.add_argument('-height', type=int)
    # parser.add_argument('-width', type=int)
    args = vars(parser.parse_args())

    main(args['root_folder'], args['root_folder'])
