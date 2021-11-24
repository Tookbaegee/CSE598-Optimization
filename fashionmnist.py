import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import wandb
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class FashionDataset(Dataset):
    """User defined class to build a datset using Pytorch class Dataset."""
    
    def __init__(self, data, transform = None):
        """Method to initilaize variables.""" 
        self.fashion_MNIST = list(data.values)
        self.transform = transform
        
        label = []
        image = []
        
        for i in self.fashion_MNIST:
             # first column is of labels.
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        # Dimension of Images = 28 * 28 * 1. where height = width = 28 and color_channels = 1.
        self.images = np.asarray(image).reshape(-1, 28, 28, 1).astype('float32')

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)

class FashionCNN(nn.Module):
    
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out

def trainWithOptimizer(model, optimizer):
    num_epochs = 50
    count = 0
    # Lists for visualization of loss and accuracy 
    loss_list = []
    iteration_list = []
    accuracy_list = []

    # Lists for knowing classwise accuracy
    predictions_list = []
    labels_list = []

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # Transfering images and labels to GPU if available
            images, labels = images.to(device), labels.to(device)

            train = Variable(images.view(100, 1, 28, 28))
            labels = Variable(labels)

            # Forward pass 
            outputs = model(train)
            loss = error(outputs, labels)

            # Initializing a gradient as 0 so there is no mixing of gradient among the batches
            optimizer.zero_grad()

            #Propagating the error backward
            loss.backward()

            # Optimizing the parameters
            optimizer.step()

            count += 1

        # Testing the model

            if not (count % 100):    # It's same as "if count % 50 == 0"
                total = 0
                correct = 0

                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    labels_list.append(labels)

                    test = Variable(images.view(100, 1, 28, 28))

                    outputs = model(test)

                    predictions = torch.max(outputs, 1)[1].to(device)
                    predictions_list.append(predictions)
                    correct += (predictions == labels).sum()

                    total += len(labels)

                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)
                # wandb.log({"loss": loss.data, "accuracy": accuracy})

            if not (count % 1000):
                print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
        # wandb.log({"loss epoch": loss_list[len(loss_list) - 1], "accuracy epoch": accuracy_list[len(accuracy_list) - 1]})

def test_model(model):
    total = 0
    correct = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        test = Variable(images.view(100, 1, 28, 28))

        outputs = model(test)
        loss = error(outputs, labels)
        print(loss.data)
        predictions = torch.max(outputs, 1)[1].to(device)
        correct += (predictions == labels).sum()

        total += len(labels)

    accuracy = correct * 100 / total

    print("loss: {}".format(loss.data) + ", accuracy: {}".format(accuracy))

if __name__ == '__main__':
    # key = "91d4844cfd842fc2c6393a72cb16c61efca9a96d"
    # wandb.init(project="optimization-final", entity="cse598qk")
    # wandb.config = {
    # "learning_rate": 0.001,
    # "epochs": 10,
    # "batch_size": 100
    # }


    train_csv = pd.read_csv("fashion-mnist_train.csv")
    test_csv = pd.read_csv("fashion-mnist_test.csv")

    train_set = FashionDataset(train_csv, transform=transforms.Compose([transforms.ToTensor()]))
    test_set = FashionDataset(test_csv, transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = DataLoader(train_set, batch_size=100)
    test_loader = DataLoader(train_set, batch_size=100)

    error = nn.CrossEntropyLoss()

    learning_rate = 0.001
    device = "cuda:0"
    model = FashionCNN()
    model.to(device)
    trainWithOptimizer(model, torch.optim.RMSprop(model.parameters(), lr=learning_rate))
    torch.save(model.state_dict(), 'fashionCNN-rms.pt')

    model = FashionCNN()
    model.to(device)
    trainWithOptimizer(model, torch.optim.Adam(model.parameters(), lr=learning_rate))
    torch.save(model.state_dict(), 'fashionCNN-adam.pt')

    model = FashionCNN()
    model.to(device)
    trainWithOptimizer(model, torch.optim.Adagrad(model.parameters(), lr=learning_rate))
    torch.save(model.state_dict(), 'fashionCNN-adagrad.pt')

    model = FashionCNN()
    model.to(device)
    trainWithOptimizer(model, torch.optim.SGD(model.parameters(), lr=learning_rate))
    torch.save(model.state_dict(), 'fashionCNN-sgd.pt')
