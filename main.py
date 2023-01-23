from data import LoadCIFAR10
train, test = LoadCIFAR10(batch_size=32)


"""
All of the result (model, loss, accuracy, plot, etc) are in the folder results
"""
from models import ResNet56A, ResNet44A, ResNet32A, ResNet20A, ResNet56B, ResNet44B, ResNet32B, ResNet20B, CNN56, CNN44, CNN32, CNN20
models = [ResNet56A(), ResNet44A(), ResNet32A(), ResNet20A(), ResNet56B(), ResNet44B(), ResNet32B(),ResNet20B(), CNN56(), CNN44(), CNN32(),CNN20()]
models_str = ["ResNet56A", "ResNet44A", "ResNet32A", "ResNet20A", "ResNet56B", "ResNet44B", "ResNet32B","ResNet20B", "CNN56", "CNN44", "CNN32","CNN20"]

from train_test_functions import global_loop
for model, model_name in zip(models, models_str):
    global_loop(model, model_name, train, test)


from models import TorchResNet50, TorchResNet34, TorchResNet18
models = [TorchResNet50(), TorchResNet34(), TorchResNet18()]
models_str = ["TorchResNet50", "TorchResNet34", "TorchResNet18"]

from train_test_functions import torch_loop
for model, model_name in zip(models, models_str):
    torch_loop(model, model_name, train, test)


from analysis import Compare_OptionA_and_B, Compare_ResNet_and_TorchResNet, Compare_ResNet_and_CNN
Compare_OptionA_and_B()
Compare_ResNet_and_TorchResNet()
Compare_ResNet_and_CNN()