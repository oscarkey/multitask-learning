import tempfile

import torch
from torch.optim import Optimizer

from model import MultitaskLearner

server_name = 'mongodb://multitask-learning:***REMOVED***@134.209.21.201/admin?retryWrites=true'
collection_name = 'multitask-learning'


def save_model(_run, model: MultitaskLearner, optimizer: Optimizer, epoch: int):
    """Saves the state of the model and optimizer to Sacred, suitable for visualisation or resuming training."""
    with tempfile.NamedTemporaryFile() as file:
        state = {'version': 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'epoch': epoch}

        torch.save(state, file.name)
        _run.add_artifact(file.name, f'model_epoch_{epoch}')
        _run.run_logger.info(f'Saved state to sacred at epoch {epoch}.')
