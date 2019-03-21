from PIL import Image
import numpy as np
import torch
from timeit import default_timer as timer
from cityscapes import CityscapesDataset

dataset = CityscapesDataset('/Users/joanna.materzynska/werk/multitask-learning/example-tiny-cityscapes/aachen')

index = 0
image_array = dataset._cached_get_image_array(index)

instance_file = dataset._get_file_path_for_index(index, 'instanceIds')
instance_array = np.asarray(Image.open(instance_file), dtype=np.float32)
start = timer()
instance_vecs, instance_mask = dataset._get_instance_vecs_and_mask(0)
print(timer() - start, "computation of centroids")
instance = {"mask": instance_mask,
            "vec": instance_vecs}
np.save('instance', instance)



import pdb

def compute_centroid_vector(instance_image):
    start = timer()
    centroids = np.zeros(instance_image.shape + (2,))
    for value in np.unique(instance_image):
        xs, ys = np.where(instance_image == value)
        centroids[xs, ys] = np.array((np.floor(np.mean(xs)), np.floor(np.mean(ys))))
    coordinates = np.zeros(instance_image.shape + (2,))
    g1, g2 = np.mgrid[range(instance_image.shape[0]), range(instance_image.shape[1])]
    coordinates[:, :, 0] = g1
    coordinates[:, :, 1] = g2
    vecs = centroids - coordinates
    mask = np.ma.masked_where(instance_image >= 1000, instance_image)
    if len(mask.mask.shape) > 1:
           mask = np.asarray(mask.mask, dtype=np.uint8)
    elif mask.mask is False:
        mask = np.zeros(instance_image.shape, dtype=np.uint8)
    else:
        mask = np.ones(instance_image.shape, dtype=np.uint8)
    mask = np.stack((mask, mask))
    print('centroids np', timer() - start)
    return vecs, mask

def compute_centroid_vector_torch(instance_image):
    start = timer()
    instance_image_tensor = torch.Tensor(instance_image)
    centroids_t = torch.zeros(instance_image.shape + (2,))
    for value in torch.unique(instance_image_tensor):
        xsys = torch.nonzero(instance_image_tensor == value)
        xs, ys = xsys[:, 0], xsys[:, 1]
        centroids_t[xs, ys] = torch.stack((torch.mean(xs.float()), torch.mean(ys.float())))

    coordinates = torch.zeros(instance_image.shape + (2,))
    g1, g2 = torch.meshgrid(torch.arange(instance_image_tensor.size()[0]), torch.arange(instance_image_tensor.size()[1]))
    coordinates[:, :, 0] = g1
    coordinates[:, :, 1] = g2
    vecs = centroids_t - coordinates
    mask = instance_image_tensor >= 1000
    if len(mask.size()) > 1:
        mask = mask.int()
    elif mask is False:
        mask = np.zeros(instance_image.shape, dtype=np.uint8)
    else:
        mask = np.ones(instance_image.shape, dtype=np.uint8)
    mask = torch.stack((mask, mask))
    print('centroids torch', timer() - start)
    return vecs, mask

for i in range(30):
    vecsnp, masknp = compute_centroid_vector(instance_array)
    vecst, maskt = compute_centroid_vector_torch(instance_array)
# print(masknp, maskt)
# print(vecsnp, vecst)