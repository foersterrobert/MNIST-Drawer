from numpy.core.defchararray import title
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os

class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

model = torch.load('./model/mnist.pth')

def transform_image(image):  # convert all images into a similar size
    transforms = T.ToTensor()
    return transforms(image)

# given the kind of input, chooses the model to predict on, returns a numpy array

def get_prediction(image_tensor):
    image_tensor = image_tensor.unsqueeze_(0)
    outputs = model(image_tensor)
    # _, predicted = torch.max(outputs.data, 1)
    return outputs.squeeze().detach().numpy()


st.set_page_config(
    page_title="MNIST-Drawer",
    page_icon=":pencil:",
)


hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title("MNIST-Drawer")


def predict(image):
    image = Image.fromarray((image[:, :, 0]).astype(np.uint8))
    image = image.resize((28, 28))
    tensor = transform_image(image)
    prediction = get_prediction(tensor)
    return prediction


def np_to_df(outputs):  # Create a 2D array for the dataframe instead of a 1D array
    length = outputs.shape[0]  # Total outputs
    arr = []
    for pos in range(0, length):
        line = [0]*10
        line[pos] = outputs[pos]
        arr.append(line)
    return arr


# Specify brush parameters and drawing mode
stroke_width = st.sidebar.slider("Stroke width: ", 1, 100, 25)


# Create a canvas component
canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color="#fff",
    background_color="#000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

result = st.button("Predict")

if canvas_result.image_data is not None and result:
    # outputs = predict(canvas_result.image_data)
    image = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
    image = image.resize((28, 28))
    tensor = transform_image(image)
    # prediction = get_prediction(tensor)
    image_tensor = tensor.unsqueeze_(0)
    image_tensor = (image_tensor - 0.1307) / 0.3081
    # image_tensor[image_tensor==0.0000] = -0.4242
    outputs = model(image_tensor)
    # _, predicted = torch.max(outputs.data, 1)
    prediction =  outputs.squeeze().detach().numpy()
    outputs = prediction
    ind_max = np.where(outputs == max(outputs))[
        0][0]  # Index of the max element
    # Converting index to equivalent letter
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)
    st.markdown("<h3 style = 'text-align: center;'>Prediction : {}<h3>".format(
        ind_max), unsafe_allow_html=True)
    chart_data = pd.DataFrame(np_to_df(outputs), index=[
                              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], columns=[
                              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    st.bar_chart(chart_data)
