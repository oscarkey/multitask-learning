"""Contains config and Sacred main entry point."""
import train
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver

ex = Experiment()

mongo = False
if mongo == True:
    mongo_observer = MongoObserver.create(
        url='mongodb+srv://multitask-learning:GJHtmxWrAvZ9pTunNAtH@cluster0-elau5.azure.mongodb.net/test?retryWrites=true',
        db_name='multitask-learning'
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
    root_dir_validation = 'example-tiny-cityscapes' # TODO: add validation set
    root_dir_test = 'example-tiny-cityscapes' # TODO: add test set
    num_classes = 20
    height = 128  # TODO: pass through to model
    width = 256  # TODO: pass through to model
    # One of 'fixed' or 'learned'.
    loss_type = 'fixed'
    loss_weights = (0.5, 0.5, 0.0)
    enabled_tasks = (True, True, False)
    gpu = False
    mongo = False


@ex.named_config
def server_config():
    gpu = True


@ex.automain
def main(_run):
    # TODO: train, then test or whatever
    train.main(_run)
