from models import (
    ResNet56A,
    ResNet44A,
    ResNet32A,
    ResNet20A,
    ResNet56B,
    ResNet44B,
    ResNet32B,
    ResNet20B,
    CNN56,
    CNN44,
    CNN32,
    CNN20,
)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

accuracy20 = [0.3769, 0.526, 0.6142, 0.6616,
              0.7067, 0.731, 0.741, 0.7539, 0.7586, 0.7631]
accuracy32 = [0.3132, 0.3846, 0.4694, 0.5292,
              0.5705, 0.6175, 0.6479, 0.6586, 0.6544, 0.6922]
accuracy44 = [0.2635, 0.3393, 0.394, 0.4454,
              0.48, 0.5162, 0.5827, 0.5861, 0.5916, 0.6233]
accuracy56 = [0.2424, 0.333, 0.356, 0.4101,
              0.4567, 0.5483, 0.5773, 0.6067, 0.6454, 0.6657]

resnet_32 = [0.5179, 0.6923, 0.7398, 0.7806,
             0.7856, 0.7957, 0.8014, 0.8209, 0.8194, 0.8271]
resnet_20 = [0.5865, 0.7112, 0.7448, 0.7733,
             0.7868, 0.8046, 0.8123, 0.8088, 0.8206, 0.801]
resnet_44 = [0.537, 0.6854, 0.7362, 0.7777,
             0.7914, 0.7962, 0.8129, 0.8241, 0.8188, 0.8152]
resnet_56 = [0.4975, 0.6792, 0.7226, 0.7677,
             0.7859, 0.8044, 0.81, 0.8243, 0.8216, 0.8262]

resnet = [resnet_20, resnet_32, resnet_44, resnet_56]

accuracyCNN = [accuracy20, accuracy32, accuracy44, accuracy56]


# Plot graphique
st.title("Residual learning")

fig = plt.figure(figsize=(16, 9))

CNNrange = [20, 32, 44, 56]

choice = st.select_slider("Choose the number of layer", options=CNNrange)


fig = plt.figure()
plt.plot(accuracyCNN[(choice-20)//12], label="CNN")
plt.plot(resnet[(choice-20)//12], label="Resnet")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title(
    "Accuracy du modèle sur le jeu de données test en fonction du nombre d'epochs")
st.pyplot(plt)

# Number of parameters

models = [["ResNet56A", ResNet56A(), "red"],
          ["ResNet44A", ResNet44A(), "green"],
          ["ResNet32A", ResNet32A(), "blue"],
          ["ResNet20A", ResNet20A(), "purple"],
          ["ResNet56B", ResNet56B(), "orange"],
          ["ResNet44B", ResNet44B(), "yellow"],
          ["ResNet32B", ResNet32B(), "brown"],
          ["ResNet20B", ResNet20B(), "pink"],
          ["CNN56", CNN56(), "black"],
          ["CNN44", CNN44(), "cyan"],
          ["CNN32", CNN32(), "gray"]]

fig, ax = plt.subplots()

space = 0
space_y = 0
center_y = 0
max_rayon = 0

for string, model, color in models:
    path = './results/' + string
    model.load_state_dict(torch.load(path))
    model.eval()

    number_param = sum(p.numel() for p in model.parameters())
    rayon = np.sqrt(number_param/np.pi)
    if rayon > max_rayon:
        max_rayon = rayon

    if string[0] == "C":
        if center_y == 0:
            center_y = -3*max_rayon

        if space_y != 0:
            space_y += 1.5*rayon

        circle = plt.Circle((space_y, center_y), rayon,
                            color=color, label=string)
        space_y += 1.5*rayon

    else:
        if space != 0:
            space += 1.5*rayon
        circle = plt.Circle((space, 0), rayon, color=color, label=string)
        space += 1.5*rayon
    ax.add_artist(circle)
    ax.legend(loc='best')

plt.xlim(-2*rayon, 1.3*space + 2*rayon)
plt.ylim(-5*max_rayon, 2*max_rayon)
ax.set_aspect(1)


st.pyplot(plt)
