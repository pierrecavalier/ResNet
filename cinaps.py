import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
#from pytorch_model_summary import summary
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 32
train = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test = DataLoader(test_data, batch_size=batch_size, shuffle=True)

print("Image shape of a random sample image : {}".format(
    training_data[0][0].numpy().shape), end='\n\n')
print("Training Set:   {} images".format(len(training_data)))
print("Test Set:       {} images".format(len(test_data)))

# need to return an tensor in option A


class Identity(nn.Module):

    def __init__(self, lambd):
        super(Identity, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# will be use as block_type ne


class ConvBlock(nn.Module):
    '''
    ConvBlock will implement the regular ConvBlock and the shortcut block. See figure 2
    When the dimension changes between 2 blocks, the option A or B of the paper are available.
    '''

    def __init__(self, in_channels, out_channels, stride=1, option='A'):
        super(ConvBlock, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        """Implementation of option A (adding pad) and B (conv2d) in the paper for matching dimensions"""

        if stride != 1 or in_channels != out_channels:
            if option == 'A':
                # number 4 as said in the paper (4pixels are padded each side)
                pad_to_add = out_channels//4
                # padding the right and bottom of
                padding = (0, 0, 0, 0, pad_to_add, pad_to_add, 0, 0)
                self.shortcut = Identity(
                    lambda x: nn.functional.pad(x[:, :, ::2, ::2], padding))
            if option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, 2*out_channels, kernel_size=1,
                              stride=stride, padding=0, bias=False),
                    nn.BatchNorm2d(2*out_channels)
                )

    def forward(self, x):
        out = self.features(x)
        # sum it up with shortcut layer
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetNN(nn.Module):
    """ 
    ResNet-56 architecture for CIFAR-10 DataSet of shape 32*32*3
    """

    def __init__(self, block_type, num_blocks):
        super(ResNetNN, self).__init__()

        self.in_channels = 16  # soit c'est filters soit c'est feature maps

        self.conv0 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(16)
        self.block1 = self.__build_layer(
            block_type, 16, num_blocks[0], starting_stride=1)
        self.block2 = self.__build_layer(
            block_type, 32, num_blocks[1], starting_stride=2)
        self.block3 = self.__build_layer(
            block_type, 64, num_blocks[2], starting_stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # jsp à quoi ca sert
        self.relu = nn.ReLU()
        self.linear = nn.Linear(64, 10)  # final activation for classification

    def __build_layer(self, block_type, out_channels, num_blocks, starting_stride):
        # create a list of len num_blocks with the first the stride we want then follow by ones
        strides_list_for_current_block = [starting_stride]+[1]*(num_blocks-1)

        # boucle for to create a mutiple layer with de good in_channels and out_channels and the good stride defined above
        layers = []
        for stride in strides_list_for_current_block:
            layers.append(block_type(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def ResNet56():
    return ResNetNN(block_type=ConvBlock, num_blocks=[9, 9, 9])


model = ResNet56()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


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


def train_model():
    EPOCHS = 15
    train_samples_num = 50000
    train_costs, val_costs = [], []

    # Training phase.
    for epoch in range(EPOCHS):

        train_running_loss = 0
        correct_train = 0

        for inputs, labels in train:

            """ for every mini-batch during the training phase, we typically want to explicitly set the gradients 
            to zero before starting to do backpropragation """
            optimizer.zero_grad()

            # Start the forward pass
            prediction = model(inputs)

            loss = criterion(prediction, labels)

            # do backpropagation and update weights with step()
            loss.backward()
            optimizer.step()

            # print('outputs on which to apply torch.max ', prediction)
            # find the maximum along the rows, use dim=1 to torch.max()
            _, predicted_outputs = torch.max(prediction.data, 1)

            # Update the running corrects
            correct_train += (predicted_outputs == labels).float().sum().item()

            ''' Compute batch loss
            multiply each average batch loss with batch-length. 
            The batch-length is inputs.size(0) which gives the number total images in each batch. 
            Essentially I am un-averaging the previously calculated Loss '''
            train_running_loss += (loss.data.item() * inputs.shape[0])

        train_epoch_loss = train_running_loss / train_samples_num

        train_costs.append(train_epoch_loss)

        train_acc = correct_train / train_samples_num

        info = "[Epoch {}/{}]: train-loss = {:0.6f} | train-acc = {:0.3f}"

        test_samples_num = 10000
        correct = 0

        with torch.no_grad():
            for inputs_test, labels_test in test:
                # Make predictions.
                prediction = model(inputs_test)

                # Retrieve predictions indexes.
                _, predicted_class = torch.max(prediction.data, 1)

                # Compute number of correct predictions.
                correct += (predicted_class ==
                            labels_test).float().sum().item()

        test_accuracy = correct / test_samples_num
        print('Test accuracy: {}'.format(test_accuracy))

        string = str(test_accuracy) + ", "
        fichier.write(string)

        print(info.format(epoch+1, EPOCHS, train_epoch_loss, train_acc))

    return train_costs


fichier = open("error.txt", "x")

train_costs = train_model()

fichier.close()

test_samples_num = 10000
correct = 0


with torch.no_grad():
    for inputs, labels in test:
        # Make predictions.
        prediction = model(inputs)

        # Retrieve predictions indexes.
        _, predicted_class = torch.max(prediction.data, 1)

        # Compute number of correct predictions.
        correct += (predicted_class == labels).float().sum().item()

test_accuracy = correct / test_samples_num
print('Test accuracy: {}'.format(test_accuracy))
