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
    mongo_observer = MongoObserver.create(
        url=checkpointing.server_name,
        db_name=checkpointing.collection_name
    )
    ex.observers.append(mongo_observer)
else:
    ex.observers.append(FileStorageObserver.create('multitask_results'))


@ex.config
def config():
    """Contains the default config values."""
    batch_size = 3
    max_iter = 1000
    root_dir_train = 'example-tiny-cityscapes'
    root_dir_validation = 'example-tiny-cityscapes'  # TODO: add validation set
    root_dir_test = 'example-tiny-cityscapes'  # TODO: add test set
    num_classes = 20
    initial_learning_rate = 2.5e-3
    height = 128  # TODO: pass through to model
    width = 256  # TODO: pass through to model
    # One of 'fixed' or 'learned'.
    loss_type = 'fixed'
    loss_weights = (1.0, 1.0, 0.0)
    enabled_tasks = (True, True, False)
    gpu = True
    save_to_db = True
    # How frequently to run validation. Set to 0 to disable validation.
    validate_epochs = 1
    # How frequently to checkpoint the model to Sacred. Set to 0 to disable saving the model.
    model_save_epochs = 0
    use_adam = False
    # If num workers > 0 then dataloader caching won't work.
    dataloader_workers = 0
    # Whether to randomly crop the training data, only works when training on full size images.
    random_crop_train = False
    crop_size = (256, 256)

@ex.named_config
def server_config():
    gpu = True
    root_dir_train = '/home/aml8/tiny_cityscapes_train'
    root_dir_validation = '/home/aml8/tiny_cityscapes_train'
    root_dir_test = '/home/aml8/tiny_cityscapes_train'
    dataloader_workers = 6


@ex.automain
def main(_run):
    # TODO: train, then test or whatever
    train.main(_run)
