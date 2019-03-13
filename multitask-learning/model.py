import torch
import torch.nn as nn

from encoder import Encoder
from decoders import Decoders

class MultitaskLearner(nn.Module):
    def __init__(self, num_classes, loss_weights):
        super(MultitaskLearner, self).__init__()
        self.encoder = Encoder()
        self.decoders = Decoders(num_classes)

        self.sem_log_var = nn.Parameter(torch.tensor(loss_weights[0], dtype=torch.float))
        self.inst_log_var = nn.Parameter(torch.tensor(loss_weights[1], dtype=torch.float))
        self.depth_log_var = nn.Parameter(torch.tensor(loss_weights[2], dtype=torch.float))

    def forward(self, x):
        """Returns sem_seg_output, instance_seg_output, depth_output"""
        return self.decoders(self.encoder(x))

    def get_loss_params(self):
        """Returns sem_log_var, inst_log_var, depth_log_var"""
        return self.sem_log_var, self.inst_log_var, self.depth_log_var