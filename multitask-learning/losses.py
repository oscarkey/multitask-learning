"""Contains the loss functions."""

import torch
from torch import nn


class MultiTaskLoss(nn.Module):
    """Computes and combines the losses for the three tasks.

    Has two modes:
    1) 'fixed': the losses multiplied by fixed weights and summed
    2) 'learned': we learn the losses, not implemented...
    """

    def __init__(self, loss_type, loss_weights=None):
        """Creates a new instance.

        :param loss_type Either 'fixed' or 'learned'
        :param loss_weights If 'fixed' then a 3 tuple (semantic seg weight, instance seg weight,
        depth weight), if 'learned' then None.
        """
        super().__init__()

        assert ((loss_type == 'learned' and loss_weights is None)
                or loss_type == 'fixed' and len(loss_weights) == 3)

        self.loss_type = loss_type
        self.loss_weights = loss_weights

        self.l1_loss = nn.L1Loss(size_average=True)
        # Classes that we don't care about are set to 255.
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255)

    def semantic_segmentation_loss(self, semseg_input, semseg_target):
        return self.cross_entropy(semseg_input, semseg_target)

    def instance_segmentation_loss(self, instance_input, instance_target, instance_mask):
        instance_mask = instance_mask.byte()
        masked_target = torch.masked_select(instance_target, instance_mask)

        masked_input = torch.masked_select(instance_input, instance_mask)
        return self.l1_loss(masked_input, masked_target.float())

    def depth_loss(self, depth_input, depth_target):
        return 0

    def calculate_total_loss(self, *losses):
        if self.loss_type == 'fixed':
            return sum([loss * weight for loss, weight in zip(losses, self.loss_weights)])

        elif self.loss_type == 'learned':
            raise NotImplementedError

        else:
            raise ValueError

    def forward(self, predicted, *target):
        semseg_pred, instance_pred, depth_pred = predicted
        # MISSING depth target
        semseg_target, instance_target, instance_mask = target

        semseg_loss = self.semantic_segmentation_loss(semseg_pred, semseg_target)
        instanceseg_loss = self.instance_segmentation_loss(instance_pred,
                                                           instance_target,
                                                           instance_mask)
        depth_loss = self.depth_loss(depth_pred, 0)

        return self.calculate_total_loss(semseg_loss, instanceseg_loss, depth_loss)
