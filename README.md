# ResNet
Project as part of the "AI Methods" course taught by [Guillermo Durand](https://durandg12.github.io/) in the first year of the [Mathematics and Artificial Intelligence master](https://www.imo.universite-paris-saclay.fr/fr/etudiants/masters/mathematiques-et-applications/m1/mathematiques-et-intelligence-artificielle/)'s program. 

Study of residual networks based on the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015).

<p align="center">
<img src="https://github.com/pierrecavalier/ResNet/blob/main/docs/resnet.png" width="300">
  </p>

## Experiment

We have implemented residual networks with a number of layers ranging from ten to fifty, trying out two options (A and B, described in the article) and determining their accuracy on a subset of the CIFAR-10 dataset.

<p align="center">
<img src="https://github.com/pierrecavalier/ResNet/blob/main/results/OptionAandB.png" width="300">
<img src=https://github.com/pierrecavalier/ResNet/blob/main/results/ResNetandTorchResNet.png width="300">
</p>


## Use


Use "streamlit run streamlit.py" to run the website.

One can vizualise, train the chosen model by clicking respectively on the buttons "View model" and "Train model".

The training of the model is done on a subset of 5000 data of CIFAR10 on 5 epochs which should take about 3-5 minutes.

Once all the desired model trained, one can compare their accuracy by clicking on the last button, "Print the accuracy of every model trained so far".
