import tempfile
from typing import Dict, Tuple

import pymongo
import torch
from torch.optim import Optimizer

from cityscapestask.model import MultitaskLearner

server_name = 'mongodb://multitask-learning:***REMOVED***@134.209.21.201/admin?retryWrites=true'
collection_name = 'multitask-learning'


def save_model(_run, model: MultitaskLearner, optimizer: Optimizer, epoch: int, iterations: int):
    """Saves the state of the model and optimizer to Sacred, suitable for visualisation or resuming training."""
    with tempfile.NamedTemporaryFile() as file:
        state = {'version': 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'epoch': epoch, 'iterations': iterations}

        torch.save(state, file.name)
        _run.add_artifact(file.name, 'model_epoch_{}'.format(epoch))
        _run.run_logger.info('Saved model to sacred at epoch {}.'.format(epoch))


def load_state(_run, run_id: int) -> Tuple[int, Dict, Dict]:
    """Loads the state of the latest save from the given run.

    :return (epoch: int, state_dict)
    """
    db = pymongo.MongoClient(server_name, 27017)[collection_name]
    experiment = db['runs'].find_one({'_id': run_id})

    artifacts = experiment['artifacts']
    _run.run_logger.debug('Found {} saves'.format(len(artifacts)))
    artifact = artifacts[len(artifacts) - 1]

    # Parse the name above: model_epoch_{epoch}
    epoch = int(artifact['name'].split('_')[2])

    cursor = db['fs.chunks'].find({'files_id': artifact['file_id']})
    with tempfile.NamedTemporaryFile() as file:
        _run.run_logger.info('Found {} chunks for epoch {}, downloading...'.format(cursor.count(), epoch))
        i = 0
        for chunk in cursor:
            assert chunk['n'] == i
            i += 1
            file.write(chunk['data'])
        _run.run_logger.debug('Download complete')
        state = torch.load(file.name)

    # We don't know how to handle anything except version 1.
    assert state['version'] == 1

    return state['epoch'], state['model_state_dict'], state['optimizer_state_dict']
