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
import keras
import pickle

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
        return x

@st.cache
def loadModels():
    PytorchModel = torch.load('./model/mnist.pth')

    for name, param in PytorchModel.named_parameters():
        if name in ['fc.weight', 'fc.bias']:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # for name, param in PytorchModel.named_parameters():
    #     print(name, ':', param.requires_grad)
    return PytorchModel
        


st.set_page_config(
    page_title="MNIST-Drawer",
    page_icon=":pencil:",
)

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title("MNIST-Drawer :pencil:")

KerasModel = keras.models.load_model('./model/Keras.pth')
PytorchModel = loadModels()
ScikitModel = pickle.load(open('./model/scikit-learn.sav', 'rb'))

with st.sidebar:
    stroke_width = st.slider("Stroke width: ", 1, 100, 25)
    framework = st.selectbox("Model:", options=['Pytorch', 'Keras', 'scikit-learn'])
    st.markdown("---")
    st.markdown(
            """
            <h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://robertfoerster.com/">Robert</a></h6>
            <br>
            <a href="https://github.com/foersterrobert/MNIST-Drawer" target='_blank'><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Logo.png" alt="Streamlit logo" height="20"></a>
            <a href="https://www.linkedin.com/in/rfoerster/" target='_blank' style='margin-left: 10px;'><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/LinkedIn_Logo.svg/1000px-LinkedIn_Logo.svg.png" alt="Streamlit logo" height="26"></a>
            """,
            unsafe_allow_html=True,
        )

def np_to_df(outputs):
    length = outputs.shape[0]
    arr = []
    for pos in range(0, length):
        line = [0]*10
        line[pos] = outputs[pos]
        arr.append(line)
    return arr

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
    image = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
    image = image.resize((28, 28))
    
    if framework == 'Pytorch':
        transforms = T.ToTensor()
        tensor = transforms(image)
        image_tensor = tensor.unsqueeze_(0)
        image_tensor = (image_tensor - 0.1307) / 0.3081
        outputs = PytorchModel(image_tensor)
        prediction =  outputs.squeeze().detach().numpy()
        outputs = prediction
        ind_max = np.where(outputs == max(outputs))[0][0]

    elif framework == 'Keras':
        array = np.array(image)
        array = np.reshape(array, (1, 28, 28, 1))
        outputs = KerasModel.predict(array).squeeze()
        ind_max = np.where(outputs == max(outputs))[0][0]

    elif framework == 'scikit-learn':
        array = np.array(image)
        array = np.reshape(array, (1, 784))
        array = (array - 33.39449141308309) / 78.6590439631829
        outputs = ScikitModel.predict_proba(array).squeeze()
        ind_max = np.where(outputs == max(outputs))[0][0]
    
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

accuracies = {
    'Pytorch': [98.7, 128, 20],
    'Keras': [99.12, 128, 15],
    'scikit-learn': [96.78, 0, 0]
     }

st.markdown("---")
st.subheader(f'Model: {framework} | Test-Accuracy: {accuracies[framework][0]}%')
if framework == 'Pytorch':
    st.write(f'Batchsize: {accuracies[framework][1]}, Epochs: {accuracies[framework][2]}')
    st.code(
        '''
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

        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)
        '''
    )

elif framework == 'Keras':
    st.write(f'Batchsize: {accuracies[framework][1]}, Epochs: {accuracies[framework][2]}')
    st.code(
        '''
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
        '''
    )

elif framework == 'scikit-learn':
    st.write('Support Vector Classification')
    st.code(
        '''
        clf = svm.SVC(gamma=0.001, probability=True)
        clf.fit(X_train, y_train)
        '''
    )

st.markdown("---")

st.markdown(
            """
            <div style='display:flex; align-items: center; justify-content: center; gap: 5px;'>
                <h3>All code on </h3>
                <a href="https://github.com/foersterrobert/MNIST-Drawer" target='_blank'><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Logo.png" alt="Streamlit logo" height="18"></a>
            </div>
            """,
            unsafe_allow_html=True,
        )