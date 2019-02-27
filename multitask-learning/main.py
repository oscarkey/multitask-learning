"""Contains config and Sacred main entry point."""
import train
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment()

mongo_observer = MongoObserver.create(
    url='mongodb+srv://multitask-learning:GJHtmxWrAvZ9pTunNAtH@cluster0-elau5.azure.mongodb.net/test?retryWrites=true',
    db_name='multitask-learning'
)
ex.observers.append(mongo_observer)


@ex.config
def config():
    """Contains the default config values."""
    batch_size = 8
    max_iter = 1000
    root_dir = 'tiny_cityscapes_train'
    num_classes = 20
    height = 128  # TODO: pass through to model
    width = 256  # TODO: pass through to model
    loss_type = 'fixed'
    loss_weights = (1, 0, 0)


@ex.automain
def main(_run):
    # TODO: train, then test or whatever
    train.train(_run)
