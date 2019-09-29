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
    def get_raw_loss(self, output: Tensor, labels: Tensor, original: Tensor) -> Tensor:
        """Return the unweighted loss, where the implementing class defines the exact loss function.

        :param output of the model
        :param labels for classification losses
        :param original The original image, for reconstruction losses
        """
        pass

    @abstractmethod
    def weight_loss(self, loss: Tensor) -> Tensor:
        """Weights the given loss appropriately. e.g. by a fixed weight or a learned uncertainty weight"""
        pass


class FixedCELoss(MnistLossFunc):
    def __init__(self, weight: float, class_map_func):
        assert isinstance(weight, float)
        self._weight = weight
        self._class_map_func = class_map_func

    def get_raw_loss(self, output: Tensor, labels: Tensor, _) -> Tensor:
        return F.cross_entropy(output, self._class_map_func(labels))

    def weight_loss(self, loss: Tensor) -> Tensor:
        return self._weight * loss


class LearnedCELoss(MnistLossFunc):
    def __init__(self, s: nn.Parameter, class_map_func):
        assert isinstance(s, nn.Parameter)
        self._s = s
        self._class_map_func = class_map_func

    def get_raw_loss(self, output: Tensor, labels: Tensor, _) -> Tensor:
        return F.cross_entropy(output, self._class_map_func(labels))

    def weight_loss(self, loss: Tensor) -> Tensor:
        return torch.exp(-self._s) * loss + 0.5 * self._s


class FixedL1Loss(MnistLossFunc):
    def __init__(self, weight: float):
        assert isinstance(weight, float)
        self._weight = weight

    def get_raw_loss(self, output: Tensor, _, original: Tensor) -> Tensor:
        return F.l1_loss(output, original)

    def weight_loss(self, loss: Tensor) -> Tensor:
        return self._weight * loss


class LearnedL1Loss(MnistLossFunc):
    def __init__(self, s: nn.Parameter):
        assert isinstance(s, nn.Parameter)
        self._s = s

    def get_raw_loss(self, output: Tensor, _, original: Tensor) -> Tensor:
        return F.l1_loss(output, original)

    def weight_loss(self, loss: Tensor) -> Tensor:
        return 0.5 * torch.exp(-self._s) * loss + 0.5 * self._s


class MultitaskMnistLoss(ABC):
    def __init__(self, enabled_tasks: [bool], loss_funcs: [MnistLossFunc]):
        super().__init__()
        assert len(enabled_tasks) == len(loss_funcs), f'enabled_tasks={enabled_tasks}, loss_funcs={loss_funcs}'
        self._enabled_tasks = enabled_tasks
        self._loss_funcs = loss_funcs

    def __call__(self, outputs: [Tensor], labels: Tensor, original: Tensor):
        """Returns (overall loss, [task losses])"""
        assert len(outputs) == len(self._enabled_tasks) == len(self._loss_funcs)

        raw_losses = [
            loss_func.get_raw_loss(output, labels, original) if enabled else torch.tensor([0.0], device=output.device)
            for enabled, loss_func, output in zip(self._enabled_tasks, self._loss_funcs, outputs)]

        weighted_losses = [loss_func.weight_loss(raw_loss) for loss_func, raw_loss in zip(self._loss_funcs, raw_losses)]
        total_loss = weighted_losses[0] + weighted_losses[1] + weighted_losses[2]

        return total_loss, (raw_losses[0], raw_losses[1], raw_losses[2])


def get_fixed_loss(enabled_tasks: [bool], weights: [float], mnist_type: str):
    """Returns the fixed weight loss function."""
    task_1_loss_func = FixedCELoss(weights[0], lambda labels: _labels_to_1(labels, mnist_type))
    task_2_loss_func = FixedCELoss(weights[1], lambda x: x)
    task_3_loss_func = FixedL1Loss(weights[2])
    return MultitaskMnistLoss(enabled_tasks, [task_1_loss_func, task_2_loss_func, task_3_loss_func])


def get_learned_loss(enabled_tasks: [bool], ses: [nn.Parameter], mnist_type: str):
    """Returns the learned uncertainties loss function.

    :param ses s=log(sigma^2) for each task, as in the paper
    """
    task_1_loss_func = LearnedCELoss(ses[0], lambda labels: _labels_to_1(labels, mnist_type))
    task_2_loss_func = LearnedCELoss(ses[1], lambda x: x)
    task_3_loss_func = LearnedL1Loss(ses[2])
    return MultitaskMnistLoss(enabled_tasks, [task_1_loss_func, task_2_loss_func, task_3_loss_func])
