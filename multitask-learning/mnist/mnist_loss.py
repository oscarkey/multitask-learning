"""The loss function for the MNIST variant of the multitask experiment."""
from abc import abstractmethod, ABC

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _labels_to_1(labels, mnist_type: str):
    """Returns labels for task 1. We classify three way: class a, class b and other. a and b depend on the task."""
    # 2 is class other.
    converted = torch.full_like(labels, 2)
    if mnist_type == 'numbers':
        # 3 and 7 are notoriously hard to distinguish between.
        converted[labels == 3] = 0
        converted[labels == 7] = 1

    elif mnist_type == 'fashion_pullover_coat':
        # Try and distinguish between pullovers (2) and coats (4), which are very similar.
        converted[labels == 2] = 0
        converted[labels == 4] = 1

    elif mnist_type == 'fashion_tshirt_shirt':
        # Try and distinguish between tshirts (0) and shirts (6).
        converted[labels == 0] = 0
        converted[labels == 6] = 1

    else:
        raise ValueError(f'Unknown mnist_type: {mnist_type}')

    return converted


def _compute_num_correct(preds: Tensor, labels: Tensor) -> int:
    assert preds.shape == labels.shape
    return (labels == preds).sum().item()


def compute_num_correct_task1(preds: Tensor, labels: Tensor, mnist_type: str) -> int:
    """Computes the number of correct predictions on a single batch for task 1"""
    return _compute_num_correct(preds, _labels_to_1(labels, mnist_type))


def compute_num_correct_task2(preds: Tensor, labels: Tensor) -> int:
    """Computes the number of correct predictions on a single batch for task 2"""
    return _compute_num_correct(preds, labels)


class MnistLossFunc(ABC):
    @abstractmethod
    def __call__(self, output: Tensor, labels: Tensor) -> Tensor:
        pass


class FixedCELoss(MnistLossFunc):
    def __init__(self, weight: float, class_map_func):
        self._weight = weight
        self._class_map_func = class_map_func

    def __call__(self, output: Tensor, labels: Tensor) -> Tensor:
        return self._weight * F.cross_entropy(output, self._class_map_func(labels))


class LearnedCELoss(MnistLossFunc):
    def __init__(self, s: nn.Parameter, class_map_func):
        assert isinstance(s, nn.Parameter)
        self._s = s
        self._class_map_func = class_map_func

    def __call__(self, output: Tensor, labels: Tensor) -> Tensor:
        loss = F.cross_entropy(output, self._class_map_func(labels))
        return torch.exp(-self._s) * loss + 0.5 * self._s


class MultitaskMnistLoss(ABC):
    def __init__(self, enabled_tasks: [bool], loss_funcs: [MnistLossFunc]):
        super().__init__()
        assert len(enabled_tasks) == len(loss_funcs), f'enabled_tasks={enabled_tasks}, loss_funcs={loss_funcs}'
        self._enabled_tasks = enabled_tasks
        self._loss_funcs = loss_funcs

    def __call__(self, outputs: [Tensor], labels: Tensor):
        """Returns (overall loss, [task losses])"""
        assert len(outputs) == len(self._enabled_tasks) == len(self._loss_funcs)

        losses = [loss_func(output, labels) if enabled else torch.tensor([0.0], device=output.device) for
                  enabled, loss_func, output in zip(self._enabled_tasks, self._loss_funcs, outputs)]

        return losses[0] + losses[1], (losses[0], losses[1])


def get_fixed_loss(enabled_tasks: [bool], weights: [float], mnist_type: str):
    """Returns the fixed weight loss function."""
    task_1_loss_func = FixedCELoss(weights[0], lambda labels: _labels_to_1(labels, mnist_type))
    task_2_loss_func = FixedCELoss(weights[1], lambda x: x)
    return MultitaskMnistLoss(enabled_tasks, [task_1_loss_func, task_2_loss_func])


def get_learned_loss(enabled_tasks: [bool], ses: [nn.Parameter], mnist_type: str):
    """Returns the learned uncertainties loss function.

    :param ses s=log(sigma^2) for each task, as in the paper
    """
    task_1_loss_func = LearnedCELoss(ses[0], lambda labels: _labels_to_1(labels, mnist_type))
    task_2_loss_func = LearnedCELoss(ses[1], lambda x: x)
    return MultitaskMnistLoss(enabled_tasks, [task_1_loss_func, task_2_loss_func])
