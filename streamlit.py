import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import plotly.express as px


accuracy20 = [
    0.3769,
    0.526,
    0.6142,
    0.6616,
    0.7067,
    0.731,
    0.741,
    0.7539,
    0.7586,
    0.7631,
]
accuracy32 = [
    0.3132,
    0.3846,
    0.4694,
    0.5292,
    0.5705,
    0.6175,
    0.6479,
    0.6586,
    0.6544,
    0.6922,
]
accuracy44 = [
    0.2635,
    0.3393,
    0.394,
    0.4454,
    0.48,
    0.5162,
    0.5827,
    0.5861,
    0.5916,
    0.6233,
]
accuracy56 = [
    0.2424,
    0.333,
    0.356,
    0.4101,
    0.4567,
    0.5483,
    0.5773,
    0.6067,
    0.6454,
    0.6657,
]

resnet_32 = [
    0.5179,
    0.6923,
    0.7398,
    0.7806,
    0.7856,
    0.7957,
    0.8014,
    0.8209,
    0.8194,
    0.8271,
]
resnet_20 = [
    0.5865,
    0.7112,
    0.7448,
    0.7733,
    0.7868,
    0.8046,
    0.8123,
    0.8088,
    0.8206,
    0.801,
]
resnet_44 = [
    0.537,
    0.6854,
    0.7362,
    0.7777,
    0.7914,
    0.7962,
    0.8129,
    0.8241,
    0.8188,
    0.8152,
]
resnet_56 = [
    0.4975,
    0.6792,
    0.7226,
    0.7677,
    0.7859,
    0.8044,
    0.81,
    0.8243,
    0.8216,
    0.8262,
]

resnet = [resnet_20, resnet_32, resnet_44, resnet_56]

accuracyCNN = [accuracy20, accuracy32, accuracy44, accuracy56]

fig = plt.figure(figsize=(16, 9))

CNNrange = [20, 32, 44, 56]

choice = st.select_slider("Choose the number of layer", options=CNNrange)


fig = plt.figure()
plt.plot(accuracyCNN[(choice - 20) // 12], label="CNN")
plt.plot(resnet[(choice - 20) // 12], label="Resnet")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title(
    "Accuracy du modèle sur le jeu de données test en fonction du nombre d'epochs"
)
st.pyplot(plt)


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

df = pd.DataFrame(columns=["Models", "# params"])
models = [
    ResNet56A(),
    ResNet44A(),
    ResNet32A(),
    ResNet20A(),
    CNN56(),
    CNN44(),
    CNN32(),
]
models_str = [
    "ResNet56",
    "ResNet44",
    "ResNet32",
    "ResNet20",
    "CNN56",
    "CNN44",
    "CNN32",
]
for model, model_str in zip(models, models_str):
    model.load_state_dict(
        torch.load(
            "./results/{}".format(model_str)
            + ("A" if model_str.startswith("ResNet") else "")
        )
    )
    append = pd.DataFrame(
        [[model_str, sum(p.numel() for p in model.parameters())]],
        columns=["Models", "# params"],
    )
    df = pd.concat([df, append], ignore_index=True)
st.write(df)

models_option = df["Models"].unique().tolist()
models_selectbox = st.multiselect(
    "Which model do you want to see ?", models_option, models_option
)
print(df)
df_temp = df[df["Models"] == models_selectbox].copy()
print(df_temp)

fig = px.scatter(
    df_temp,
    x="Models",
    y=np.zeros(len(df_temp["# params"])),
    size=df_temp["# params"].tolist(),
    color="# params",
)
st.write(fig)
