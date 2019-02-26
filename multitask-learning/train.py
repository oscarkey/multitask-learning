from decoders import Decoders
from encoder import Encoder

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


if __name__ == '__main__':

    root_dir='tiny_cityscapes_train'

    loader = cityscapes.get_loader_from_dir(root_dir, config={'batch_size': 3})

    learner = MultitaskLearner(num_classes=20)

    # TODO: polynomial lr decay
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    max_iter = 2
    initial_learning_rate = 2.5e-3

    for epoch in range(max_iter):  # loop over the dataset multiple times
        learning_rate = initial_learning_rate*(1-epoch/max_iter)**0.9
        optimizer = torch.optim.SGD(learner.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=1e4)

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