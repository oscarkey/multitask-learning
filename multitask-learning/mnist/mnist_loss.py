"""The loss function for the MNIST variant of the multitask experiment."""
from abc import abstractmethod, ABC

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _labels_to_1(labels):
    """Returns labels adapted for task 1. Classes are is 3 -> 0, 7->0, other->3."""
    # 2 is class other.
    converted = torch.full_like(labels, 2)
    converted[labels == 3] = 0
    converted[labels == 7] = 1
    return converted


def _compute_loss1(output, labels):
    return F.cross_entropy(output, _labels_to_1(labels))


def _compute_loss2(output, labels):
    return F.cross_entropy(output, labels)


def _compute_num_correct(preds: Tensor, labels: Tensor) -> int:
    assert preds.shape == labels.shape
    return (labels == preds).sum().item()


def compute_num_correct_task1(preds: Tensor, labels: Tensor) -> int:
    """Computes the number of correct predictions on a single batch for task 1"""
    return _compute_num_correct(preds, _labels_to_1(labels))


def compute_num_correct_task2(preds: Tensor, labels: Tensor) -> int:
    """Computes the number of correct predictions on a single batch for task 2"""
    return _compute_num_correct(preds, labels)


class MultitaskMnistLoss(ABC):
    @abstractmethod
    def __call__(self, output1: Tensor, output2: Tensor, labels) -> (Tensor, (float, float)):
        """Returns (overall loss, (loss task 1, loss task 2))"""
        pass


class FixedWeightsLoss(MultitaskMnistLoss):
    def __init__(self, enable_task1: bool, enable_task2: bool, weight1: float, weight2: float):
        super().__init__()
        assert isinstance(weight1, float)
        assert isinstance(weight2, float)
        self._enable_task1 = enable_task1
        self._enable_task2 = enable_task2
        self._weight1 = weight1
        self._weight2 = weight2

    def __call__(self, output1: Tensor, output2: Tensor, labels) -> (Tensor, (Tensor, Tensor)):
        if self._enable_task1:
            loss1 = _compute_loss1(output1, labels)
        else:
            output1
            loss1 = torch.tensor([0], dtype=torch.float, device=output1.device)

        if self._enable_task2:
            loss2 = _compute_loss2(output2, labels)
        else:
            loss2 = torch.tensor([0], dtype=torch.float, device=output1.device)

        return self._weight1 * loss1 + self._weight2 * loss2, (loss1, loss2)


class LearnedWeightsLoss(MultitaskMnistLoss):
    def __init__(self, weight1: nn.Parameter, weight2: nn.Parameter):
        super().__init__()
        assert isinstance(weight1, nn.Parameter)
        assert isinstance(weight2, nn.Parameter)
        self._weight1 = weight1
        self._weight2 = weight2

    def __call__(self, output1: Tensor, output2: Tensor, labels) -> (Tensor, (Tensor, Tensor)):
        loss1 = _compute_loss1(output1, labels)
        loss2 = _compute_loss2(output2, labels)
        loss = (torch.exp(-self._weight1) * loss1 + 0.5 * self._weight1 + torch.exp(
            -self._weight2) * loss2 + 0.5 * self._weight2)
        return loss, (loss1, loss2)
