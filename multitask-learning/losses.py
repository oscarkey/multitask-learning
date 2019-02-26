from torch import nn

class MultiTaskLoss(nn.Module):

    def __init__(self):
        pass

    def semantic_segmentation_loss(self):
        return nn.CrossEntropyLoss()

    def instance_segmentation_loss(self):
        pass

    def depth_loss(self):
        pass

    def forward(self, input, *target):
        semseg_loss = self.semantic_segmentation_loss(input[0], target[0])
        instanceseg_loss = self.instance_segmentation_loss(input[1], target[1])
        depth_loss = self.depth_loss(input[2], target[2])
        total_loss = None # weight uncertanity
        return total_loss