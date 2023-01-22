import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
#from pytorch_model_summary import summary
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

batch_size = 32
train = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# needed to return an tensor in option A
class Identity(nn.Module):

    def __init__(self, lambd):
        super(Identity, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# will be use as block_type next
class ConvBlock(nn.Module):
    '''
    ConvBlock will implement the regular ConvBlock and the shortcut block. See figure 2
    When the dimension changes between 2 blocks, the option A or B of the paper are available.
    '''

    def __init__(self, in_channels, out_channels, stride=1, option='A', ResNet = True):
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
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=stride, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
        self.ResNet = ResNet

    def forward(self, x):
        out = self.features(x)
        # sum it up with shortcut layer
        if self.ResNet:
            out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetNN(nn.Module):
    """ 
    ResNet-56 architecture for CIFAR-10 DataSet of shape 32*32*3
    """

    def __init__(self, block_type, num_blocks, option, ResNet):
        super(ResNetNN, self).__init__()

        self.in_channels = 16  # soit c'est filters soit c'est feature maps

        self.conv0 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(16)
        self.block1 = self.__build_layer(
            block_type, 16, num_blocks[0], starting_stride=1, option=option, ResNet=ResNet)
        self.block2 = self.__build_layer(
            block_type, 32, num_blocks[1], starting_stride=2, option=option, ResNet=ResNet)
        self.block3 = self.__build_layer(
            block_type, 64, num_blocks[2], starting_stride=2, option=option, ResNet=ResNet)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # jsp à quoi ca sert
        self.relu = nn.ReLU()
        self.linear = nn.Linear(64, 10)  # final activation for classification

    def __build_layer(self, block_type, out_channels, num_blocks, starting_stride, option, ResNet):
        # create a list of len num_blocks with the first the stride we want then follow by ones
        strides_list_for_current_block = [starting_stride]+[1]*(num_blocks-1)

        # boucle for to create a mutiple layer with de good in_channels and out_channels and the good stride defined above
        layers = []
        for stride in strides_list_for_current_block:
            layers.append(block_type(self.in_channels, out_channels, stride, option, ResNet))
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


def ResNet56A():
    return ResNetNN(block_type=ConvBlock, num_blocks=[9, 9, 9], option='A', ResNet=True)

def ResNet44A():
    return ResNetNN(block_type=ConvBlock, num_blocks=[7,7,7], option='A', ResNet=True)

def ResNet32A():
    return ResNetNN(block_type=ConvBlock, num_blocks=[5,5,5], option='A', ResNet=True)

def ResNet20A():
    return ResNetNN(block_type=ConvBlock, num_blocks=[3,3,3], option='A', ResNet=True)

def ResNet56B():
    return ResNetNN(block_type=ConvBlock, num_blocks=[9, 9, 9], option='B', ResNet=True)

def ResNet44B():
    return ResNetNN(block_type=ConvBlock, num_blocks=[7,7,7], option='B', ResNet=True)

def ResNet32B():
    return ResNetNN(block_type=ConvBlock, num_blocks=[5,5,5], option='B', ResNet=True)

def ResNet20B():
    return ResNetNN(block_type=ConvBlock, num_blocks=[3,3,3], option='B', ResNet=True)

def CNN56():
    return ResNetNN(block_type=ConvBlock, num_blocks=[9, 9, 9], option='A', ResNet=False)

def CNN44():
    return ResNetNN(block_type=ConvBlock, num_blocks=[7,7,7], option='A', ResNet=False)

def CNN32():
    return ResNetNN(block_type=ConvBlock, num_blocks=[5,5,5], option='A', ResNet=False)

def CNN20():
    return ResNetNN(block_type=ConvBlock, num_blocks=[3,3,3], option='A', ResNet=False)

models = [ResNet56A(), ResNet44A(), ResNet32A(), ResNet20A(), ResNet56B(), ResNet44B(), ResNet32B(),ResNet20B(), CNN56(), CNN44(), CNN32(),CNN20()]
models_str = ["ResNet56A", "ResNet44A", "ResNet32A", "ResNet20A", "ResNet56B", "ResNet44B", "ResNet32B","ResNet20B", "CNN56", "CNN44", "CNN32","CNN20"]

def train_model(model, model_name):
    EPOCHS = 10
    train_samples_num = 50000
    train_costs, val_costs = [], []

    print(model_name)

    # Training phase.
    for epoch in range(EPOCHS):

        train_running_loss = 0
        correct_train = 0

        for inputs, labels in train:

            inputs.to(device)
            labels.to(device)

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
        fichier = open("{}.txt".format(model_name), "w")
        fichier.write(string)
        fichier.close()
        torch.save(model.state_dict(), './models/{}'.format(model_name))


        print(info.format(epoch+1, EPOCHS, train_epoch_loss, train_acc))



    return train_costs


for model, model_name in zip(models, models_str):

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_costs = train_model(model, model_name)