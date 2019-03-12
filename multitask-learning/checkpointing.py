import tempfile
from typing import Tuple, Dict

import pymongo
import torch

server_name = 'mongodb://multitask-learning:***REMOVED***@134.209.21.201/admin?retryWrites=true'
collection_name = 'multitask-learning'


def save_model(_run, model, epoch: int):
    """Saves the model to sacred."""
    with tempfile.NamedTemporaryFile() as file:
        torch.save(model.state_dict(), file.name)
        _run.add_artifact(file.name, f'model_epoch_{epoch}')
        _run.run_logger.info(f'Saved model to sacred at epoch {epoch}.')
