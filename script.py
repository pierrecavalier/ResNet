import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import numpy as np

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


def train_loop(dataloader, model, loss_fn, optimizer, prin=False):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation (always in three steps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0 and prin:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return 100*(1-correct)


class RSNeuralNetwork(nn.Module):
    def __init__(self):
        super(RSNeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten() # Flatten inputs

        self.start = self.layer(1, 32, 64)

        self.layer1 = self.layer(64, 64, 64)
        self.layer2 = self.layer(64, 64, 64)
        self.layer3 = self.layer(64, 64, 64)

        self.output = self.output_stack(28)

    def layer(self, in_, between, out):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_,
                out_channels=between,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=between,
                out_channels=out,
                kernel_size=5,
                stride=1,
                padding=2, ),
        )

    def output_stack(self, size):
        return nn.Sequential(
            nn.Linear(64 * size * size, 10)
        )

    def forward(self, x):
        #x = self.flatten(x)

        logits = self.start(x)
        logits = self.layer1(logits)
        logits = self.layer2(logits)
        logits = self.layer3(logits)

        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        logits = logits.view(logits.size(0), -1)
        logits = self.output(logits)
        return logits


model = RSNeuralNetwork()


# Hyperparameters
learning_rate = 0.01
batch_size = 256
epochs = 5
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Training
error_w_7 = []
for t in range(epochs):
    print(f"Epoch {t+1}-----------------")
    # Use train_loop and test_loop functions
    learning_rate = 0.1/(10*t+1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loop(train_dataloader, model, loss_fn, optimizer, prin=True)
    x = test_loop(test_dataloader, model, loss_fn)
    error_w_7.append(x)
print("Done!")

PATH = './cifar_net.pth'
torch.save(model(), PATH)
