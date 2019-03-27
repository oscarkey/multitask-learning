"""Contains the loss functions."""
from typing import Union

import torch
from torch import Tensor
from torch import nn


class MultiTaskLoss(nn.Module):
    """Computes and combines the losses for the three tasks.

    Has two modes:
    1) 'fixed': the losses multiplied by fixed weights and summed
    2) 'learned': we learn the losses, not implemented...
    """

    def __init__(self, loss_type, loss_uncertainties, enabled_tasks=(True, True, True)):
        """Creates a new instance.

        :param loss_type Either 'fixed' or 'learned'
        :param loss_uncertainties A 3 tuple of (semantic seg uncertainty, instance seg uncertainty,
        depth uncertainty). If 'fixed' then these should be floats, if 'learned' then they should be
        torch Parameters.
        """
        super().__init__()

        assert len(loss_uncertainties) == 3
        assert len(enabled_tasks) == 3
        assert ((loss_type == 'learned' and isinstance(loss_uncertainties[0], nn.parameter.Parameter)) or (
                loss_type == 'fixed' and isinstance(loss_uncertainties[0], float)))

        self.loss_type = loss_type
        self.loss_uncertainties = loss_uncertainties
        self.enabled_tasks = enabled_tasks

        self.l1_loss = nn.L1Loss(reduction='sum')

        # Classes that we don't care about are set to 255.
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255)

    def sem_seg_loss(self, sem_seg_input, sem_seg_target):
        return self.cross_entropy(sem_seg_input, sem_seg_target)

    def inst_seg_loss(self, instance_input, instance_target, instance_mask):
        instance_mask = instance_mask.float()

        target = instance_target.float() * instance_mask
        mult_loss = self.l1_loss(instance_input * instance_mask, target)
        num_nonzero = torch.nonzero(target).size(0)
        if num_nonzero > 0:
            mult_loss /= num_nonzero
        else:
            mult_loss = torch.zeros_like(mult_loss)

        return mult_loss

    def depth_loss(self, depth_input, depth_target, depth_mask):
        depth_mask = depth_mask.float()
        target = depth_target.float() * depth_mask
        mult_loss = self.l1_loss(depth_input * depth_mask, target)
        num_nonzero = torch.nonzero(target).size(0)
        if num_nonzero > 0:
            mult_loss /= num_nonzero
        else:
            mult_loss = torch.zeros_like(mult_loss)

        return mult_loss

    def calculate_total_loss(self, *losses):
        sem_loss, inst_loss, depth_loss = losses
        sem_uncertainty, inst_uncertainty, depth_uncertainty = self.loss_uncertainties
        sem_enabled, inst_enabled, depth_enabled = self.enabled_tasks

        loss = 0

        if self.loss_type == 'fixed':
            if sem_enabled:
                loss += sem_uncertainty * sem_loss
            if inst_enabled:
                loss += inst_uncertainty * inst_loss
            if depth_enabled:
                loss += depth_uncertainty * depth_loss

        elif self.loss_type == 'learned':
            if sem_enabled:
                loss += torch.exp(-sem_uncertainty) * sem_loss + 0.5 * sem_uncertainty
            if inst_enabled:
                loss += 0.5 * (torch.exp(-inst_uncertainty) * inst_loss + inst_uncertainty)
            if depth_enabled:
                loss += 0.5 * (torch.exp(-depth_uncertainty) * depth_loss + depth_uncertainty)

        else:
            raise ValueError

        return loss

    def forward(self, predicted, *target) -> (Union[Tensor, None], (float, float, float)):
        sem_seg_pred, instance_pred, depth_pred = predicted
        sem_seg_target, instance_target, instance_mask, depth_target, depth_mask = target

        sem_enabled, inst_enabled, depth_enabled = self.enabled_tasks
        sem_seg_loss = self.sem_seg_loss(sem_seg_pred, sem_seg_target) if sem_enabled else None
        inst_seg_loss = self.inst_seg_loss(instance_pred, instance_target, instance_mask) if inst_enabled else None
        depth_loss = self.depth_loss(depth_pred, depth_target, depth_mask) if depth_enabled else None

        total_loss = self.calculate_total_loss(sem_seg_loss, inst_seg_loss, depth_loss)

        sem_seg_loss_item = sem_seg_loss.item() if sem_seg_loss is not None else 0
        inst_seg_loss_item = inst_seg_loss.item() if inst_seg_loss is not None else 0
        depth_loss_item = depth_loss.item() if depth_loss is not None else 0

        return total_loss, (sem_seg_loss_item, inst_seg_loss_item, depth_loss_item)
