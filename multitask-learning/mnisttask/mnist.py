import sys
import tempfile
from logging import Logger

import sacred
import torch
import torch.nn.functional as F
import torchvision
from sacred.arg_parser import get_config_updates
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver
from torch.utils.data import DataLoader
from torchvision import transforms

from mnisttask import mnist_loss
from mnisttask.mnist_loss import MultitaskMnistLoss
from mnisttask.mnist_model import MultitaskMnistModel

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
    # Whether to use the standard MNIST or FashionMNIST dataset, and so what type of classifcation task to perform.
    # See mnist_loss._labels_to_1()
    mnist_type = 'numbers'
    max_epochs = 100
    lr = 0.0001
    weight_decay = 0
    batch_size = 64
    # One of 'learned' or 'fixed'.
    loss_type = 'fixed'
    enabled_tasks = (True, False, False)
    weights = (1.0, 1.0, 1.0)
    initial_ses = (1.0, 1.0, 1.0)
    save_to_db = True
    # When True, will save a copy of the model to sacred at the end of training.
    checkpoint_at_end = False
    model_version = 1


@ex.capture
def _get_dataloaders(mnist_type: str, batch_size: int) -> (DataLoader, DataLoader):
    if mnist_type == 'numbers':
        # Where did these normalisation numbers come from????
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        model_dir = '~/.torch/models/mnist'
        train_dataset = torchvision.datasets.MNIST(model_dir, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(model_dir, train=False, download=True, transform=transform)

    elif mnist_type in ('fashion_pullover_coat', 'fashion_tshirt_shirt'):
        # TODO: normalize?
        transform = transforms.Compose([transforms.ToTensor()])
        model_dir = '~/.torch/models/fashion_mnist'
        train_dataset = torchvision.datasets.FashionMNIST(model_dir, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(model_dir, train=False, download=True, transform=transform)

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
        return _get_learned_loss_func(model=model)
    else:
        raise ValueError(f'Unknown loss type: {loss_type}')


@ex.capture
def _get_fixed_loss_func(enabled_tasks: [bool], weights: [float], mnist_type: str):
    return mnist_loss.get_fixed_loss(enabled_tasks, weights, mnist_type)


@ex.capture
def _get_learned_loss_func(enabled_tasks: [bool], model: MultitaskMnistModel, mnist_type: str):
    return mnist_loss.get_learned_loss(enabled_tasks, model.get_loss_weights(), mnist_type)


@ex.capture
def _get_model(initial_ses: [float], model_version: int) -> MultitaskMnistModel:
    return MultitaskMnistModel(initial_ses, model_version)


@ex.capture
def _get_optimizer(model: MultitaskMnistModel, lr: float, weight_decay: float):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def _get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@ex.capture
def _validate(test_dataloader: DataLoader, model: MultitaskMnistModel, mnist_type: str,
              loss_func: MultitaskMnistLoss) -> ((float, float, float), (float, float, float)):
    """Returns ((accuracy1, accuracy2, accuracy3), (loss1, loss2, loss3)).

    accuracy1 and accuracy2 are the fraction of the images which the model labelled correctly. accuracy3 is the mean
    reconstruction error.
    """
    with torch.no_grad():
        task_1_num_correct = 0
        task_2_num_correct = 0
        task_3_accum_error = 0
        task_1_accum_loss = 0
        task_2_accum_loss = 0
        task_3_accum_loss = 0
        num_images = 0
        num_batches = 0

        for data in test_dataloader:
            image, labels = data

            image = image.to(_get_device())
            labels = labels.to(_get_device())

            output1, output2, output3 = model(image)

            _, (loss1, loss2, loss3) = loss_func([output1, output2, output3], labels, image)

            task_1_accum_loss += loss1.item()
            task_2_accum_loss += loss2.item()
            task_3_accum_loss += loss3.item()

            preds1 = output1.argmax(dim=1)
            preds2 = output2.argmax(dim=1)
            assert preds1.shape == preds2.shape

            task_1_num_correct += mnist_loss.compute_num_correct_task1(preds1, labels, mnist_type)
            task_2_num_correct += mnist_loss.compute_num_correct_task2(preds2, labels)
            num_images += preds1.shape[0]

            task_3_accum_error += F.l1_loss(output3, image).sum().item()
            num_batches += 1

    assert isinstance(task_1_accum_loss, float)
    assert isinstance(task_2_accum_loss, float)
    assert isinstance(task_3_accum_loss, float)

    accuracies = task_1_num_correct / num_images, task_2_num_correct / num_images, task_3_accum_error / num_batches
    losses = task_1_accum_loss / num_batches, task_2_accum_loss / num_batches, task_3_accum_loss / num_batches
    return accuracies, losses


def _save_model(_run, model: MultitaskMnistModel):
    with tempfile.NamedTemporaryFile() as file:
        state = {'version': 1, 'model_state_dict': model.state_dict()}

        torch.save(state, file.name)
        _run.add_artifact(file.name, 'model_end')
        _run.run_logger.info('Saved model to sacred')


@ex.capture
def _train(_run, max_epochs: int, _log: Logger, checkpoint_at_end: bool):
    train_dataloader, test_dataloader = _get_dataloaders()

    model = _get_model()
    model = model.to(_get_device())

    loss_func = _get_loss_func(model=model)
    optimizer = _get_optimizer(model=model)

    _log.info('Starting training...')

    for epoch in range(max_epochs):
        epoch_loss = 0
        epoch_loss1 = 0
        epoch_loss2 = 0
        epoch_loss3 = 0

        iteration_count = 1
        for i, data in enumerate(train_dataloader):
            images, labels = data

            images = images.to(_get_device())
            labels = labels.to(_get_device())

            optimizer.zero_grad()

            outputs = model(images)

            loss, (loss1, loss2, loss3) = loss_func(outputs, labels, images)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()
            epoch_loss3 += loss3.item()

            iteration_count += 1

        weight1, weight2, weight3 = model.get_loss_weights()
        _log.info(f'Epoch {epoch}: {epoch_loss / iteration_count:.3f} '
                  f'({weight1.item():.3f}, {weight2.item():.3f}, {weight3.item():.3f})')

        (acc1, acc2, acc3), (val_loss1, val_loss2, val_loss3) = _validate(test_dataloader=test_dataloader, model=model,
                                                                          loss_func=loss_func)

        _run.log_scalar('train_loss', epoch_loss / iteration_count, epoch)
        _run.log_scalar('train_loss1', epoch_loss1 / iteration_count, epoch)
        _run.log_scalar('train_loss2', epoch_loss2 / iteration_count, epoch)
        _run.log_scalar('train_loss3', epoch_loss3 / iteration_count, epoch)
        _run.log_scalar('val_loss1', val_loss1, epoch)
        _run.log_scalar('val_loss2', val_loss2, epoch)
        _run.log_scalar('val_loss3', val_loss3, epoch)
        _run.log_scalar('val_acc1', acc1, epoch)
        _run.log_scalar('val_acc2', acc2, epoch)
        _run.log_scalar('val_acc3', acc3, epoch)
        _run.log_scalar('weight1', weight1.item(), epoch)
        _run.log_scalar('weight2', weight2.item(), epoch)
        _run.log_scalar('weight3', weight3.item(), epoch)

    if checkpoint_at_end:
        _save_model(_run, model)


@ex.automain
def main(_run):
    _train()
