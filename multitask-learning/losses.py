import torch
from torch import nn

class MultiTaskLoss(nn.Module):

    def __init__(self):
        self.l1_loss = nn.L1Loss(size_average=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def semantic_segmentation_loss(self, semseg_input, semseg_target):
        return self.cross_entropy(semseg_input, semseg_target)

    def instance_segmentation_loss(self, instance_input, instance_target, instance_mask):
        masked_input = torch.masked_selected(instance_input, instance_mask)
        masked_target = torch.masked_selected(instance_target, instance_mask)
        return self.l1_loss(masked_input, masked_target) 

    def depth_loss(self):
        pass

    def forward(self, input, *target):
        semseg_input, instance_input, depth_input = input
        # MISSING depth target
        semseg_target, instance_target, instance_mask = target
        semseg_loss = self.semantic_segmentation_loss(semseg_input, semseg_target)
        instanceseg_loss = self.instance_segmentation_loss(instance_input, instance_target, instance_mask)
        depth_loss = self.depth_loss(depth_input, depth_target)
        total_loss = None # weight uncertanity
        return total_loss
