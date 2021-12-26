import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
import time
import torch
from torchvision import transforms
from tensorflow import keras
import pickle
from pytorchTrain import PytorchDrawer
from GANpyTorchTrain import Generator

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

@st.experimental_singleton
def load_models():
    PytorchModel = torch.load('./model/Pytorch.pth')
    PytorchModel.eval()
    ScikitModel = pickle.load(open('./model/scikit-learn.sav', 'rb'))
    PytorchGenerator = Generator(100, 1, 28)
    PytorchGenerator.load_state_dict(torch.load('./model/PytorchDCGAN.pth',
                        map_location=torch.device('cpu')))
    PytorchGenerator.eval()
    return PytorchModel, ScikitModel, PytorchGenerator

KerasModel = keras.models.load_model('./model/Keras.pth')
PytorchModel, ScikitModel, PytorchGenerator = load_models()

with st.sidebar:
    page = st.radio("Page: ", ("Draw", "Generate"))
    framework = st.selectbox(
        "Prediction-Model:", options=['Pytorch', 'Keras', 'scikit-learn'])
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
    if 'noise' not in st.session_state or generate:
        st.session_state['noise'] = torch.randn(1, 100, 1, 1)
    fake = PytorchGenerator(st.session_state.noise)
    fake = fake.reshape(28, 28, 1).detach().numpy().squeeze()
    fake = (fake - np.min(fake))/np.ptp(fake)
    st.image(fake, width=280)

result = st.button(f"Predict with {framework}")

if result:
    if page == "Draw":
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

if page == "Generate":
    st.markdown("---")
    st.subheader('DCGAN Generator Architecture')
    st.code(
        """
class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x channels_noise | 1 x 100
            self._block(100, features_g * 32, 7, 1, 0),  # img: 7x7x896
            self._block(features_g * 32, features_g * 16, 4, 2, 1),  # img: 14x14x448
            self._block(features_g * 16, features_g * 8, 3, 1, 1),  # img: 14x14x224
            self._block(features_g * 8, features_g * 4, 3, 1, 1),  # img: 14x14x112
            nn.ConvTranspose2d(
                features_g * 4, 1, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img | 28x28x1
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)
        """
    )

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