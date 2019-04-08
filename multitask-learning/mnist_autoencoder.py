import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# class Model(nn.Module):
#     def __init__(self, size: (int, int), num_classes: int, batchnorm, weight_init=[1.0, 1.0]):
#         super().__init__()
#         self.size = size
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
#             nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
#             nn.Tanh()
#         )
#         self.classifier = Classifier(num_classes=10)
#         self.weight1 = nn.Parameter(torch.tensor([weight_init[0]]))
#         self.weight2 = nn.Parameter(torch.tensor([weight_init[1]]))
        
#     def forward(self, x):
#         x = self.encoder(x)
#         clss = self.classifier(x)
#         gen = self.decoder(x)
#         return clss, gen
    
# class Classifier(nn.Module):
#     def __init__(self, num_classes: int):
#         super().__init__()
#         self.fc1 = nn.Linear(in_features=32, out_features=128)
#         self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        
#     def forward(self, x):
# #         assert_shape(x, (4, 4))
        
#         x = x.view(x.shape[0], -1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class Model(nn.Module):
    def __init__(self, size: (int, int), num_classes: int, batchnorm, weight_init=[1.0, 1.0]):
        super().__init__()
        self.size = size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(64, 32, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )
        self.classifier = Classifier(num_classes=10)
        self.weight1 = nn.Parameter(torch.tensor([weight_init[0]]))
        self.weight2 = nn.Parameter(torch.tensor([weight_init[1]]))
        
    def forward(self, x):
        x = self.encoder(x)
        clss = self.classifier(x)
        gen = self.decoder(x)
        return clss, gen
    
class Classifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features=128, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
        
    def forward(self, x):
#         assert_shape(x, (4, 4))
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(train_dataloader, num_epochs, model, criterion1, criterion2, optimizer,
          enable, learn_weights, fixed_weights_vals, file_name, resume=False):
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
                w_0.append(np.round(model.weight1.detach().cpu().numpy(), 5))
                w_1.append(np.round(model.weight2.detach().cpu().numpy(), 5))
                loss1_log.append(np.round(l1, 5))
                loss2_log.append(np.round(l2, 5))               
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
    if resume:
        file_name = file_name + '_resumed'
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
        print('Reconstruction error: {}'.format(total_loss/(i+1)))
    return 100 * correct1/total, total_loss/(i+1)





def run(train_dataloader, enable, learn_weights, weights_vals, file_name,
        num_epochs, batchnorm=False):
    print('running {}'.format(file_name))
    resume = False
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.L1Loss()
    model1 = Model((28,28), 10, batchnorm, weights_vals)
    model1 = model1.cuda()
    if os.path.exists(RES_DIR + file_name):
        model1.load_state_dict(torch.load(RES_DIR + file_name))
        print('resuming ...')
        resume = True
    optimizer = torch.optim.Adam(model1.parameters(), lr=0.001, weight_decay=0.0001)

    train(train_dataloader, num_epochs, model1, criterion1, criterion2, optimizer, 
      enable, learn_weights, weights_vals, file_name, resume=resume)
    acc, l = evaluate(test_dataloader, model1, criterion2)





if __name__ == "__main__":
    num_epochs = 100

    dataset = sys.argv[1]

    run_single = True
    run_fixed = False
    run_learned = True

    if dataset == 'fashionmnist':
        RES_DIR = 'fashion_mnist_model2/'

        if not os.path.exists(RES_DIR):
            os.makedirs(RES_DIR)
        bnorm = True
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = torchvision.datasets.FashionMNIST('~/.torch/models/fashionmnist', train=True, download=True, 
                                                   transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST('~/.torch/models/fashionmnist', train=False, download=True, 
                                                   transform=transform)
    if dataset == 'mnist':
        bnorm = False
        RES_DIR = 'mnist_exp_new_model2/'
        if not os.path.exists(RES_DIR):
            os.makedirs(RES_DIR)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = torchvision.datasets.MNIST('~/.torch/models/mnist', train=True, download=True, 
                                                   transform=transform)
        test_dataset = torchvision.datasets.MNIST('~/.torch/models/mnist', train=False, download=True, 
                                                   transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    if run_single:
        # single tasks
        run(train_dataloader, enable=(True, False), learn_weights=False, 
            weights_vals=[1., 0.], batchnorm=bnorm, file_name='classification_only', num_epochs=num_epochs)
        run(train_dataloader, enable=(False, True), learn_weights=False, 
            weights_vals=[0., 1.], batchnorm=bnorm, file_name='reconstruction_only', num_epochs=num_epochs)
    if run_fixed:
        # fixed grid search
        run(train_dataloader, enable=(True, True), learn_weights=False, 
            weights_vals=[0.1, 0.9], batchnorm=bnorm, file_name='fixed_0.1_0.9', num_epochs=num_epochs)

        run(train_dataloader, enable=(True, True), learn_weights=False, 
            weights_vals=[0.2, 0.8], batchnorm=bnorm, file_name='fixed_0.2_0.8', num_epochs=num_epochs)

        run(train_dataloader, enable=(True, True), learn_weights=False, 
            weights_vals=[0.3, 0.7], batchnorm=bnorm, file_name='fixed_0.3_0.7', num_epochs=num_epochs)

        run(train_dataloader, enable=(True, True), learn_weights=False, 
            weights_vals=[0.4, 0.6], batchnorm=bnorm, file_name='fixed_0.4_0.6', num_epochs=num_epochs)

        run(train_dataloader, enable=(True, True), learn_weights=False, 
            weights_vals=[0.5, 0.5], batchnorm=bnorm, file_name='fixed_0.5_0.5', num_epochs=num_epochs)

        run(train_dataloader, enable=(True, True), learn_weights=False, 
            weights_vals=[0.6, 0.4], batchnorm=bnorm, file_name='fixed_0.6_0.4', num_epochs=num_epochs)

        run(train_dataloader, enable=(True, True), learn_weights=False, 
            weights_vals=[0.7, 0.3], batchnorm=bnorm, file_name='fixed_0.7_0.3', num_epochs=num_epochs)

        run(train_dataloader, enable=(True, True), learn_weights=False, 
            weights_vals=[0.8, 0.2], batchnorm=bnorm, file_name='fixed_0.8_0.2', num_epochs=num_epochs)

        run(train_dataloader, enable=(True, True), learn_weights=False, 
            weights_vals=[0.9, 0.1], batchnorm=bnorm, file_name='fixed_0.9_0.1', num_epochs=num_epochs)

        # # fixed grid search
        run(train_dataloader, enable=(True, True), learn_weights=False, 
            weights_vals=[0.15, 0.85], batchnorm=bnorm, file_name='fixed_0.15_0.85', num_epochs=num_epochs)

        run(train_dataloader, enable=(True, True), learn_weights=False, 
            weights_vals=[0.25, 0.75], batchnorm=bnorm, file_name='fixed_0.25_0.75', num_epochs=num_epochs)

        run(train_dataloader, enable=(True, True), learn_weights=False, 
            weights_vals=[0.005, 0.995], batchnorm=bnorm, file_name='fixed_0.005_0.995', num_epochs=num_epochs)

    if run_learned:
       # learned
        run(train_dataloader, enable=(True, True), learn_weights=True, 
            weights_vals=[0.0, 0.0], batchnorm=bnorm, file_name='learned_init_0_0', num_epochs=num_epochs)

        run(train_dataloader, enable=(True, True), learn_weights=True, 
            weights_vals=[1.0, 1.0], batchnorm=bnorm, file_name='learned_init_1_1', num_epochs=num_epochs)

        run(train_dataloader, enable=(True, True), learn_weights=True, 
            weights_vals=[0.5, 0.5], batchnorm=bnorm, file_name='learned_init_05_05', num_epochs=num_epochs)

        run(train_dataloader, enable=(True, True), learn_weights=True, 
            weights_vals=[2.0, 2.0], batchnorm=bnorm, file_name='learned_init_2_2', num_epochs=num_epochs)

        run(train_dataloader, enable=(True, True), learn_weights=True, 
            weights_vals=[-0.5, -0.5], batchnorm=bnorm, file_name='learned_init_05_05neg', num_epochs=num_epochs)



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
