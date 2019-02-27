"""Contains config and Sacred main entry point."""
from sacred import Experiment
import train

ex = Experiment()

@ex.config
def config():
    """Contains the default config values."""
    batch_size = 8
    max_iter = 1000
    root_dir='tiny_cityscapes_train'
    num_classes=20
    height = 128 #TODO: pass through to model
    width = 256 #TODO: pass through to model


@ex.automain
def main(_run):
    # TODO: train, then test or whatever
    train.train(_run)
