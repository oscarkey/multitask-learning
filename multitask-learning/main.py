"""Contains config and Sacred main entry point."""
import sys

from sacred import Experiment
from sacred.arg_parser import get_config_updates
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

import checkpointing
import train

ex = Experiment()

config_updates, _ = get_config_updates(sys.argv)

# Disable saving to mongo using "with save_to_db=False"
if ("save_to_db" not in config_updates) or config_updates["save_to_db"]:
    mongo_observer = MongoObserver.create(url=checkpointing.server_name, db_name=checkpointing.collection_name)
    ex.observers.append(mongo_observer)
else:
    ex.observers.append(FileStorageObserver.create('multitask_results'))


@ex.config
def config():
    """Contains the default config values."""
    batch_size = 3
    max_iter = 1000
    root_dir_train = 'example-tiny-cityscapes'
    root_dir_validation = 'example-tiny-cityscapes'
    root_dir_test = 'example-tiny-cityscapes'
    num_classes = 20
    initial_learning_rate = 2.5e-4
    height = 128  # TODO: pass through to model
    width = 256  # TODO: pass through to model
    loss_type = 'learned'  # One of 'fixed' or 'learned'.
    loss_uncertainties = (1.0, 1.0, 1.0)  # equal to weights when loss_type = 'fixed'
    enabled_tasks = (True, True, True)
    gpu = True
    save_to_db = True
    validate_epochs = 1  # How frequently to run validation. Set to 0 to disable validation.
    # When True, we will run one validation pass and then exit. We will not train. This is useful to validate a previous
    # experiment using restore_from_sacred_run below.
    validate_only = False
    model_save_epochs = 0  # How frequently to checkpoint the model to Sacred. Set to 0 to disable saving the model.
    # Id of the sacred run to continue training on, or -1 to disable restoring.
    restore_sacred_run = -1
    use_adam = True
    # The learning rate used by Adam. Not used by SGD.
    learning_rate = 1e-3
    # Weight decay to set on the optimizer. Value from paper is 10e4
    weight_decay = 0
    dataloader_workers = 0  # If num workers > 0 then dataloader caching won't work.
    # When True the dataloader will cache all data in memory after the first read.
    dataloader_cache = True
    # When True the data loader will load precomputed instance vectors from the .npy files.
    use_precomputed_instances = False
    # Whether to augment the training data with random cropping.
    crop = False
    crop_size = (64, 64)
    # Whether to augment the training data with random flipping.
    flip = False
    pre_train_encoder = True  # When true, will download weights for resnet pre-trained on imagenet.
    # If total available memory is lower than this threshold, we crash rather than loading more data.
    # This avoids using all the memory on the server and getting it stuck.
    # Set to 0 to disable the check.
    min_available_memory_gb = 0
    # Size of the dilations in the atrous convolutions in ASPP module of the encoder. Paper default is (12, 24, 36).
    aspp_dilations = (12, 24, 36)


@ex.named_config
def tiny_cityscapes_crops():
    """Crops of 64x64 from Tiny Cityscapes train, with random flipping, validated on Tiny Cityscales val"""
    crop = True
    flip = True
    crop_size = (64, 64)
    max_iter = 50000
    root_dir_train = '/data/home/aml8/tiny_cityscapes_train'
    root_dir_validation = '/data/home/aml8/tiny_cityscapes_val'
    root_dir_test = 'example-tiny-cityscapes'  # TODO: add test set
    batch_size = 24
    learning_rate = 2.5e-5

@ex.named_config
def tiny_cityscapes():
    """Tiny Cityscapes train, with random flipping, validated on Tiny Cityscales val"""
    crop = False
    flip = True
    max_iter = 50000
    root_dir_train = '/data/home/aml8/tiny_cityscapes_train'
    root_dir_validation = '/data/home/aml8/tiny_cityscapes_val'
    root_dir_test = 'example-tiny-cityscapes'  # TODO: add test set
    batch_size = 16
    max_iter = 50000
    use_adam = False
    initial_learning_rate = 1e-3
    weight_decay = 10e-4


@ex.named_config
def cityscapes_crops():
    """Crops of 256x256 Cityscapes, with random flipping, validated on Tiny Cityscales val"""
    crop_size = (256, 256)
    batch_size = 8
    max_iter = 50000
    root_dir_train = '/data/home/aml8/cityscapes/train'
    root_dir_validation = '/data/home/aml8/tiny_cityscapes_val'
    root_dir_test = 'example-tiny-cityscapes'  # TODO: add test set
    crop = True
    crop_size = (256,256)
    flip = True


@ex.named_config
def server_config():
    gpu = True
    root_dir_train = '/home/aml8/tiny_cityscapes_train'
    root_dir_validation = '/home/aml8/tiny_cityscapes_train'
    root_dir_test = '/home/aml8/tiny_cityscapes_train'
    dataloader_workers = 6


@ex.automain
def main(_run):
    # TODO: add testing loop
    train.main(_run)
