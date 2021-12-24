import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from torchvision import transforms
from tensorflow import keras
import pickle
from pytorchTrain import PytorchDrawer
from GANpyTorchTrain import sample_noise, Generator

def np_to_df(outputs):
    length = outputs.shape[0]
    arr = []
    for pos in range(0, length):
        line = [0]*10
        line[pos] = outputs[pos]
        arr.append(line)
    return arr

st.set_page_config(
    page_title="MNIST-Drawer & Generator",
    page_icon=":pencil:",
)

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title("MNIST-Drawer & Generator :pencil:")

KerasModel = keras.models.load_model('./model/Keras.pth')
PytorchModel = torch.load('./model/Pytorch.pth')
ScikitModel = pickle.load(open('./model/scikit-learn.sav', 'rb'))
netG = Generator(100, 1, 64)
netG.load_state_dict(torch.load('./model/PytorchGAN.pth',
                     map_location=torch.device('cpu')))

with st.sidebar:
    page = st.radio("Page: ", ("Draw", "Generate"))
    framework = st.selectbox(
        "Model:", options=['Pytorch', 'Keras', 'scikit-learn'])
    if page == 'Draw':
        stroke_width = st.slider("Stroke width: ", 1, 50, 20)
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

if page == "Draw":
    canvas = st_canvas(
        stroke_width=stroke_width,
        stroke_color="#fff",
        background_color="#000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

else:
    generate = st.button("Generate with DCGAN")
    if 'noise' not in st.session_state:
        st.session_state['noise'] = sample_noise
    if generate:
        st.session_state['noise'] = torch.randn(1, 100, 1, 1)
    fake = netG(st.session_state.noise)
    fake = fake.reshape(64, 64, 1).detach().numpy().squeeze()
    fake = (fake - np.min(fake))/np.ptp(fake)
    st.image(fake, width=280)

result = st.button(f"Predict with {framework}")

if result:
    if page == "Draw":
        if canvas.image_data != None:
            image = Image.fromarray((canvas.image_data[:, :, 0]).astype(np.uint8))
    else:
        image = Image.fromarray((fake*255).astype(np.uint8))
    image = image.resize((28, 28))

    if framework == 'Pytorch':
        DataTransform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])
        tensor = DataTransform(image)
        image_tensor = tensor.unsqueeze_(0)
        with torch.no_grad():
            outputs = torch.exp(PytorchModel(image_tensor))
            outputs = outputs.squeeze().detach().numpy() ** 0.2
            ind_max = np.where(outputs == max(outputs))[0][0]

    elif framework == 'Keras':
        array = np.array(image)
        array = array.reshape(1, 28, 28, 1)
        outputs = KerasModel.predict(array).squeeze()
        ind_max = np.where(outputs == max(outputs))[0][0]

    elif framework == 'scikit-learn':
        array = np.array(image)
        array = array.reshape(1, 784)
        array = (array - 33.385964741253645) / 78.65437362689433
        outputs = ScikitModel.predict_proba(array).squeeze()
        ind_max = np.where(outputs == max(outputs))[0][0]

    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)
    st.markdown("<h3 style = 'text-align: center;'>Prediction : {}<h3>".format(
        ind_max), unsafe_allow_html=True)
    chart_data = pd.DataFrame(np_to_df(outputs),
                              index=['0', '1', '2', '3', '4','5', '6', '7', '8', '9'],
                              columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    st.bar_chart(chart_data)

accuracies = {
    'Pytorch': [98.8, 64, 20],
    'Keras': [99.12, 128, 15],
    'scikit-learn': [96.78, 0, 0]
}

st.markdown("---")
st.subheader(
    f'Model: {framework} | Test-Accuracy: {accuracies[framework][0]}%')
if framework == 'Pytorch':
    st.write(
        f'Batchsize: {accuracies[framework][1]}, Epochs: {accuracies[framework][2]}')
    st.code(
        '''
class PytorchDrawer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        '''
    )

elif framework == 'Keras':
    st.write(
        f'Batchsize: {accuracies[framework][1]}, Epochs: {accuracies[framework][2]}')
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
            <div style='text-align: center;'>
                <a style='text-decoration: none;' href="https://github.com/foersterrobert/MNIST-Drawer" target='_blank'><h3>All code on GitHub</h3></a>
            </div>
            """,
    unsafe_allow_html=True,
)