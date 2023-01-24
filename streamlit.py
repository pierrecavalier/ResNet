import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.write("My First Streamlit Web App")

df = pd.DataFrame({"one": [1, 2, 3], "two": [4, 5, 6], "three": [7, 8, 9]})
st.write(df)

accuracy20 = [0.3769, 0.526, 0.6142, 0.6616,
              0.7067, 0.731, 0.741, 0.7539, 0.7586, 0.7631]
accuracy32 = [0.3132, 0.3846, 0.4694, 0.5292,
              0.5705, 0.6175, 0.6479, 0.6586, 0.6544, 0.6922]
accuracy44 = [0.2635, 0.3393, 0.394, 0.4454,
              0.48, 0.5162, 0.5827, 0.5861, 0.5916, 0.6233]
accuracy56 = [0.2424, 0.333, 0.356, 0.4101,
              0.4567, 0.5483, 0.5773, 0.6067, 0.6454, 0.6657]

accuracyCNN = [accuracy20, accuracy32, accuracy44, accuracy56]

fig = plt.figure(figsize=(16, 9))

CNNrange = [0, 1, 2, 3]
number = ["20", "32", "44", "56"]

choice = st.select_slider("Choose the number of layer", options=CNNrange)

string = number[choice] + "couches"
st.line_chart(accuracyCNN[choice])
