import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


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
    coordinates = np.zeros(instance_image.shape + (2,))
    g1, g2 = np.mgrid[range(instance_image.shape[0]), range(instance_image.shape[1])]
    coordinates[:, :, 0] = g1
    coordinates[:, :, 1] = g2
    vecs = centroids - coordinates
    mask = np.ma.masked_where(instance_image >= 1000, instance_image)

    # To catch instances where the mask is all false
    if len(mask.mask.shape) > 1:
        mask = np.asarray(mask.mask, dtype=np.uint8)
    elif mask.mask is False:
        mask = np.zeros(instance_image.shape, dtype=np.uint8)
    else:
        mask = np.ones(instance_image.shape, dtype=np.uint8)
    mask = np.stack((mask, mask))
    vecs = np.transpose(vecs, (2, 0, 1))
    return vecs, mask


def save_mask_instances_file(output_folder, file_path):
    instance_array = np.asarray(Image.open(file_path), dtype=np.float32)
    file_path = file_path.replace('instanceIds', 'instanceMask')
    instance_vecs, instance_mask = _compute_centroid_vectors(instance_array)
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
