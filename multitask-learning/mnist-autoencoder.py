import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

RES_DIR = 'mnist_exp/'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_dataset = torchvision.datasets.MNIST('~/.torch/models/mnist', train=True, download=True, 
                                           transform=transform)
test_dataset = torchvision.datasets.MNIST('~/.torch/models/mnist', train=False, download=True, 
                                           transform=transform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)


def assert_shape(x, shape):
    assert tuple(x.shape[-2:]) == tuple(shape), f'Expected shape ending {shape}, got {x.shape}'



class Encoder(nn.Module):
    def __init__(self, size: (int, int)):
        super().__init__()
        self.size = size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=12, padding=0, stride=2)
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=64, kernel_size=5, padding=2)

    
    def forward(self, x):
        assert_shape(x, self.size)
        
        x = F.relu(self.conv1(x))
        assert_shape(x, (9, 9)) # Should this be the same size?

        x = F.relu(self.conv2(x))
        assert_shape(x, (9, 9))
        
        x = F.max_pool2d(x, 2)
        assert_shape(x, (4, 4))
        
        return x
    
class Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features=4*4*64, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)
        
    def forward(self, x):
#         assert_shape(x, (4, 4))
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=0),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=1, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x
    
class Model(nn.Module):
    def __init__(self, size: (int, int), num_classes: int, weight_init=[1.0, 1.0]):
        super().__init__()
        self.size = size
        self.encoder = Encoder(size)
        self.autoencoder = Decoder()
        self.decoder = Classifier(num_classes=10)
        
        self.weight1 = nn.Parameter(torch.tensor([weight_init[0]]))
        self.weight2 = nn.Parameter(torch.tensor([weight_init[1]]))
        
    def forward(self, x):
#         assert_shape(x, self.size)
#         print(x.size())
        x = self.encoder(x)
#         print(x.size())
        cls_ = self.decoder(x)
#         print(cls.size())
        gen = self.autoencoder(x)
#         print(gen.size())
        return cls_, gen

def train(train_dataloader, num_epochs, model, criterion1, criterion2, optimizer,
          enable, learn_weights, fixed_weights_vals, file_name):
    info = {}
    info['fixed_weights_vals'] = fixed_weights_vals
    info['enable'] = [str(enable[0]), str(enable[1])]
    info['learn_weights'] = str(learn_weights)
    if enable[0]:
        w_0 = []
    if enable[1]:
        w_1 = []
    loss1_log, loss2_log = [], []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, data in enumerate(train_dataloader):
            image, labels = data

            image = image.cuda()
            labels = labels.cuda()

            image /= 255

            optimizer.zero_grad()

            output1, output2 = model(image)

            if enable[0]:
                loss1 = criterion1(output1, labels)
                l1 = loss1.item()
            else:
                loss1 = 0
                l1 = 0

            if enable[1]:
                loss2 = criterion2(image, output2)
                l2 = loss2.item()
            else:
                loss2 = 0
                l2 = 0

            if learn_weights:
                loss = (torch.exp(-model.weight1) * loss1 + 0.5 * model.weight1 
                        + 0.5 * (torch.exp(-model.weight2) * loss2 + model.weight2))
                w_0.append(np.round(model.weight1.detach().cpu().numpy()), 5)
                w_1.append(np.round(model.weight1.detach().cpu().numpy()), 5)
            else:
                loss = fixed_weights_vals[0]*loss1 + fixed_weights_vals[1]*loss2
                loss1_log.append(np.round(l1, 5))
                loss2_log.append(np.round(l2, 5))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch}: {epoch_loss/i:.3} ({model.weight1.item()}, {model.weight2.item()})')
    info['loss1'] = loss1_log
    info['loss2'] = loss2_log  
    info['weight1'] = w_0
    info['weight2'] = w_1
    np.save(RES_DIR + file_name, info) 
    torch.save(model.state_dict(), RES_DIR + file_name)



def evaluate(test_dataloader, model, criterion2):
    with torch.no_grad():
        correct1 = 0
        total = 0
        total_loss = 0

        for i, data in enumerate(test_dataloader):
            image, labels = data

            image = image.cuda()
            labels = labels.cuda()

            output1, gen = model(image)
            loss = criterion2(gen, image)
            preds1 = output1.argmax(dim=1)

            correct1 += (labels == preds1).sum().item()
            total += preds1.shape[0]
            total_loss += loss

        print(f'Accuracy 1: {correct1}/{total} ({100 * correct1/total:.2f}%)')
        print('Reconstruction error: {}'.format(total_loss/i))
    return 100 * correct1/total, total_loss/i



num_epochs = 1

def run(train_dataloader, enable, learn_weights, weights_vals, file_name,
        num_epochs=100):
    print('running {}'.format(file_name))
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.L1Loss()
    model1 = Model((28,28), 10, weights_vals)
    model1 = model1.cuda()
    optimizer = torch.optim.Adam(model1.parameters(), lr=0.001, weight_decay=0.0001)

    train(train_dataloader, num_epochs, model1, criterion1, criterion2, optimizer, 
      enable, learn_weights, weights_vals, file_name)


# single tasks
run(train_dataloader, enable=(True, False), learn_weights=False, 
    weights_vals=[1., 0.], file_name='classification_only')
run(train_dataloader, enable=(False, True), learn_weights=False, 
    weights_vals=[0., 1.], file_name='reconstruction_only')

# fixed grid search
run(train_dataloader, enable=(True, True), learn_weights=False, 
    weights_vals=[0.1, 0.9], file_name='fixed_0.1_0.9')

run(train_dataloader, enable=(True, True), learn_weights=False, 
    weights_vals=[0.2, 0.8], file_name='fixed_0.2_0.8')

run(train_dataloader, enable=(True, True), learn_weights=False, 
    weights_vals=[0.3, 0.7], file_name='fixed_0.3_0.7')

run(train_dataloader, enable=(True, True), learn_weights=False, 
    weights_vals=[0.4, 0.6], file_name='fixed_0.4_0.6')

run(train_dataloader, enable=(True, True), learn_weights=False, 
    weights_vals=[0.5, 0.5], file_name='fixed_0.5_0.5')

run(train_dataloader, enable=(True, True), learn_weights=False, 
    weights_vals=[0.6, 0.4], file_name='fixed_0.6_0.4')

run(train_dataloader, enable=(True, True), learn_weights=False, 
    weights_vals=[0.7, 0.3], file_name='fixed_0.7_0.3')

run(train_dataloader, enable=(True, True), learn_weights=False, 
    weights_vals=[0.8, 0.2], file_name='fixed_0.8_0.2')

run(train_dataloader, enable=(True, True), learn_weights=False, 
    weights_vals=[0.9, 0.1], file_name='fixed_0.9_0.1')

# learned
run(train_dataloader, enable=(True, True), learn_weights=True, 
    weights_vals=[1.0, 1.0], file_name='learned_init_1_1')

run(train_dataloader, enable=(True, True), learn_weights=True, 
    weights_vals=[1.0, 1.0]*2, file_name='learned_init_2_2')

run(train_dataloader, enable=(True, True), learn_weights=True, 
    weights_vals=[1.0, 1.0]*3, file_name='learned_init_3_3')

run(train_dataloader, enable=(True, True), learn_weights=True, 
    weights_vals=[1.0, 1.0]*4, file_name='learned_init_4_4')

run(train_dataloader, enable=(True, True), learn_weights=True, 
    weights_vals=[1.0, 1.0]*5, file_name='learned_init_5_5')

# Just 1:
# Accuracy 1: 9868/10000 (98.68%)
# Accuracy 2: 1352/10000 (13.52%)
# 
# Just 2:
# Accuracy 1: 1270/10000 (12.70%)
# Accuracy 2: 9434/10000 (94.34%)
# 
# Both (equal weights):
# Accuracy 1: 9840/10000 (98.40%)
# Accuracy 2: 9457/10000 (94.57%)
# 
# Both (learned weights):
# Accuracy 1: 9897/10000 (98.97%)
# Accuracy 2: 9735/10000 (97.35%)
