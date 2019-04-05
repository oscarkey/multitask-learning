import sys
from logging import Logger

import sacred
import torch
import torchvision
from sacred.arg_parser import get_config_updates
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver
from torch.utils.data import DataLoader
from torchvision import transforms

import mnist_loss
from mnist_loss import FixedWeightsLoss, MultitaskMnistLoss, LearnedWeightsLoss
from mnist_model import MultitaskMnistModel

ex = sacred.Experiment()

config_updates, _ = get_config_updates(sys.argv)

# Disable saving to mongo using "with save_to_db=False"
if ("save_to_db" not in config_updates) or config_updates["save_to_db"]:
    mongo_observer = MongoObserver.create(
        url='mongodb://multitask-learning:***REMOVED***@134.209.21.201/admin?retryWrites=true',
        db_name='multitask-learning')
    ex.observers.append(mongo_observer)
else:
    ex.observers.append(FileStorageObserver.create('multitask_results'))


@ex.config
def config():
    """Default config values."""
    # Allows us to filter to mnist results only in sacredboard.
    mnist = 1
    # One of 'numbers', 'fashion'.
    mnist_type = 'numbers'
    max_epochs = 60
    lr = 0.0001
    batch_size = 64
    # One of 'learned' or 'fixed'.
    loss_type = 'fixed'
    enable1 = False
    enable2 = True
    weight1 = 1.0
    weight2 = 1.0
    save_to_db = True


@ex.capture
def _get_dataloaders(mnist_type: str, batch_size: int) -> (DataLoader, DataLoader):
    if mnist_type == 'numbers':
        # Where did these normalisation numbers come from????
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        model_dir = '~/.torch/models/mnist'
        train_dataset = torchvision.datasets.MNIST(model_dir, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(model_dir, train=False, download=True, transform=transform)

    elif mnist_type == 'fashion':
        raise NotImplementedError

    else:
        raise ValueError(f'Unknown MNIST type: {mnist_type}')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_dataloader, test_dataloader


@ex.capture
def _get_loss_func(loss_type: str, model: MultitaskMnistModel) -> MultitaskMnistLoss:
    if loss_type == 'fixed':
        return _get_fixed_loss_func()
    elif loss_type == 'learned':
        return _get_learned_loss_func(model)
    else:
        raise ValueError(f'Unknown loss type: {loss_type}')


@ex.capture
def _get_fixed_loss_func(enable1: bool, enable2: bool, weight1: float, weight2: float):
    return FixedWeightsLoss(enable1, enable2, weight1, weight2)


@ex.capture
def _get_learned_loss_func(model: MultitaskMnistModel):
    weight1, weight2 = model.get_loss_weights()
    return LearnedWeightsLoss(weight1, weight2)


def _get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _validate(test_dataloader: DataLoader, model: MultitaskMnistModel) -> (float, float):
    """Returns (accuracy1, accuracy2)."""
    with torch.no_grad():
        correct1 = 0
        correct2 = 0
        total = 0

        for i, data in enumerate(test_dataloader):
            image, labels = data

            image = image.to(_get_device())
            labels = labels.to(_get_device())

            output1, output2 = model(image)

            preds1 = output1.argmax(dim=1)
            preds2 = output2.argmax(dim=1)
            assert preds1.shape == preds2.shape

            correct1 += mnist_loss.compute_num_correct_task1(preds1, labels)
            correct2 += mnist_loss.compute_num_correct_task2(preds2, labels)
            total += preds1.shape[0]
    return correct1 / total, correct2 / total


@ex.capture
def _train(_run, max_epochs: int, lr: float, _log: Logger):
    train_dataloader, test_dataloader = _get_dataloaders()

    model = MultitaskMnistModel()
    model = model.to(_get_device())

    loss_func = _get_loss_func(model=model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    _log.info('Starting training...')

    for epoch in range(max_epochs):
        epoch_loss = 0
        epoch_loss1 = 0
        epoch_loss2 = 0

        for i, data in enumerate(train_dataloader):
            images, labels = data

            images = images.to(_get_device())
            labels = labels.to(_get_device())

            images /= 255

            optimizer.zero_grad()

            output1, output2 = model(images)

            loss, (loss1, loss2) = loss_func(output1, output2, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()

        weight1, weight2 = model.get_loss_weights()
        _log.info(f'Epoch {epoch}: {epoch_loss / i:.3f} ({weight1.item():.3f}, {weight2.item():.3f})')

        acc1, acc2 = _validate(test_dataloader, model)

        _run.log_scalar('train_loss', epoch_loss / i, epoch)
        _run.log_scalar('train_loss1', epoch_loss1 / i, epoch)
        _run.log_scalar('train_loss2', epoch_loss2 / i, epoch)
        _run.log_scalar('val_acc1', acc1, epoch)
        _run.log_scalar('val_acc2', acc2, epoch)
        _run.log_scalar('weight1', weight1.item(), epoch)
        _run.log_scalar('weight2', weight2.item(), epoch)


@ex.automain
def main(_run):
    _train()
