from decoders import Decoders
from encoder import Encoder

import torch.nn as nn
import cityscapes 


from PIL import Image

class MultitaskLearner(nn.Module):
    def __init__(self, num_classes):
        super(MultitaskLearner, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoders(num_classes)

    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == '__main__':

    root_dir='example-tiny-cityscapes'

    loader = cityscapes.get_loader_from_dir(root_dir, config={'batch_size': 3})

    #img = Image.open(root_dir + '/aachen_000000_000019_leftImg8bit.png')
    
    learner = MultitaskLearner(num_classes=30)

    for i, data in enumerate(loader):
        img, semantic_label, s = data
        print('Result:', learner.forward(img.float()))

