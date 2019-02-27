from decoders import Decoders
from encoder import Encoder
from losses import MultiTaskLoss

import torch
import torch.nn as nn
import cityscapes 

import torchvision.transforms as transforms

from PIL import Image

class MultitaskLearner(nn.Module):
    def __init__(self, num_classes):
        super(MultitaskLearner, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoders(num_classes)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train(_run):

    loader = cityscapes.get_loader_from_dir(_run.config['root_dir'], _run.config)
    learner = MultitaskLearner(_run.config['num_classes'])

    criterion = MultiTaskLoss(_run.config['loss_type'], _run.config['loss_weights'])

    initial_learning_rate = 2.5e-3

    optimizer = torch.optim.SGD(learner.parameters(), lr=initial_learning_rate, momentum=0.9, nesterov=True, weight_decay=1e4)
    lr_lambda = lambda x: initial_learning_rate * (1 - x / _run.config['max_iter']) ** 0.9
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)



    for epoch in range(_run.config['max_iter']):  # loop over the dataset multiple times

        #polynomial learning rate decay
        lr_scheduler.step()

        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            inputs, semantic_labels, instance_labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output_semantic, output_instance, output_depth = learner(inputs.float())
            loss = criterion(output_semantic, semantic_labels.long())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            #if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0