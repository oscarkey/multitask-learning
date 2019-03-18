"""Contains the loss functions."""

import torch
from torch import nn


class MultiTaskLoss(nn.Module):
    """Computes and combines the losses for the three tasks.

    Has two modes:
    1) 'fixed': the losses multiplied by fixed weights and summed
    2) 'learned': we learn the losses, not implemented...
    """

    def __init__(self, loss_type, loss_weights, enabled_tasks=(True, True, True)):
        """Creates a new instance.

        :param loss_type Either 'fixed' or 'learned'
        :param loss_weights A 3 tuple of (semantic seg weight, instance seg weight,
        depth weight). If 'fixed' then these should be floats, if 'learned' then they should be
        torch Parameters.
        """
        super().__init__()

        assert len(loss_weights) == 3
        assert len(enabled_tasks) == 3
        assert ((loss_type == 'learned' and isinstance(loss_weights[0], nn.parameter.Parameter)) or (
                loss_type == 'fixed' and isinstance(loss_weights[0], float)))

        self.loss_type = loss_type
        self.loss_weights = loss_weights
        self.enabled_tasks = enabled_tasks

        self.l1_loss = nn.L1Loss(reduction='sum')

        # Classes that we don't care about are set to 255.
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255)

    def semantic_segmentation_loss(self, semseg_input, semseg_target):
        return self.cross_entropy(semseg_input, semseg_target)

    def instance_segmentation_loss(self, instance_input, instance_target, instance_mask):
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
        sem_weight, inst_weight, depth_weight = self.loss_weights
        sem_enabled, inst_enabled, depth_enabled = self.enabled_tasks

        loss = 0

        if self.loss_type == 'fixed':
            if sem_enabled:
                loss += sem_weight * sem_loss
            if inst_enabled:
                loss += inst_weight * inst_loss
            if depth_enabled:
                loss += depth_weight * depth_loss

        elif self.loss_type == 'learned':
            if sem_enabled:
                loss += torch.exp(-sem_weight) * sem_loss + 0.5 * sem_weight
            if inst_enabled:
                loss += 0.5 * (torch.exp(-inst_weight) * inst_loss + inst_weight)
            if depth_enabled:
                loss += 0.5 * (torch.exp(-depth_weight) * depth_loss + depth_weight)

        else:
            raise ValueError

        return loss

    def forward(self, predicted, *target):
        semseg_pred, instance_pred, depth_pred = predicted
        semseg_target, instance_target, instance_mask, depth_target, depth_mask = target

        semseg_loss = self.semantic_segmentation_loss(semseg_pred, semseg_target)
        instanceseg_loss = self.instance_segmentation_loss(instance_pred, instance_target, instance_mask)
        depth_loss = self.depth_loss(depth_pred, depth_target, depth_mask)

        total_loss = self.calculate_total_loss(semseg_loss, instanceseg_loss, depth_loss)

        return total_loss, (semseg_loss, instanceseg_loss, depth_loss)
