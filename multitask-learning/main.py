"""Contains config and Sacred main entry point."""
from sacred import Experiment

ex = Experiment()


@ex.config
def config():
    """Contains the default config values."""
    batch_size = 128


@ex.automain
def main(_run):
    config = _run.config
    # TODO: train, then test or whatever
    pass
