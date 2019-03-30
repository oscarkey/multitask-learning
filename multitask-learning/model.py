import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from decoders import Decoders
from encoder import Encoder


class MultitaskLearner(nn.Module):
    def __init__(self, num_classes, enabled_tasks: (bool, bool, bool), loss_uncertainties, pre_train_encoder,
                 output_size=(128, 256)):
        super(MultitaskLearner, self).__init__()

        encoder = Encoder()
        if pre_train_encoder:
            # Use ImageNet pre-trained weights for the ResNet-like layers of the encoder
            state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
            # strict = False so we ignore incompatibilities between official reset and our resnet.
            encoder.load_state_dict(state_dict, strict=False)
        self.encoder = encoder

        self.decoders = Decoders(num_classes, enabled_tasks, output_size)

        self.sem_log_var = nn.Parameter(torch.tensor(loss_uncertainties[0], dtype=torch.float))
        self.inst_log_var = nn.Parameter(torch.tensor(loss_uncertainties[1], dtype=torch.float))
        self.depth_log_var = nn.Parameter(torch.tensor(loss_uncertainties[2], dtype=torch.float))

    def forward(self, x):
        """Returns sem_seg_output, instance_seg_output, depth_output"""
        return self.decoders(self.encoder(x))

    def get_loss_params(self) -> (nn.Parameter, nn.Parameter, nn.Parameter):
        """Returns sem_log_var, inst_log_var, depth_log_var"""
        return self.sem_log_var, self.inst_log_var, self.depth_log_var

    def set_output_size(self, size):
        self.decoders.set_output_size(size)


if __name__ == '__main__':
    # ### Shape test
    model0 = MultitaskLearner(num_classes=20, loss_uncertainties=(1, 0, 0), pre_train_encoder=False)
    test = torch.zeros(size=(2, 3, 256, 256))
    result = model0.forward(test)
    assert result[0].shape == (2, 20, 128, 256), "output shape is {}".format(result[0].shape)

    # Check the weights have been properly loaded
    model1 = MultitaskLearner(num_classes=20, loss_uncertainties=(1, 1, 1), pre_train_encoder=True)
    model_state_dict = model1.state_dict()
    pretrained_state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
    for key in pretrained_state_dict.keys():
        model_key = 'encoder.' + str(key)
        if model_key in model_state_dict:
            assert torch.eq(pretrained_state_dict[key], model_state_dict[model_key]).all()
            print(key)
