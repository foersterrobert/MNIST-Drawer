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
from PytorchModels import PytorchDrawer, DCGAN, CGAN
from PIL import Image

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
            div.row-widget.stRadio > div{flex-direction:row; gap: 5px;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title("MNIST-Drawer & Generator :pencil:")

@st.experimental_singleton(func=None)
def load_models():
    device = torch.device('cpu')
    PytorchModel = PytorchDrawer()
    PytorchModel.load_state_dict(torch.load('./models/Pytorch.pth',
                        map_location=device))
    PytorchModel.eval()
    KerasModel = keras.models.load_model('./models/Keras')
    ScikitModel = pickle.load(open('./models/scikit-learn.sav', 'rb'))
    dcgan = DCGAN(100, 1, 28)
    dcgan.load_state_dict(torch.load('./models/DCGAN.pth',
                        map_location=device))
    dcgan.eval()
    cgan = CGAN(100, 1, 28, 10, 20)
    cgan.load_state_dict(torch.load('./models/CWGAN-GP.pth',
                        map_location=device))
    cgan.eval()
    return PytorchModel, KerasModel, ScikitModel, dcgan, cgan


PytorchModel, KerasModel, ScikitModel, dcgan, cgan = load_models()

page = st.radio("Page: ", ("Draw", "Generate"))
with st.sidebar:
    framework = st.selectbox(
        "Prediction-Model:", options=['Pytorch', 'Keras', 'scikit-learn'])
    if page == 'Draw':
        stroke_width = st.slider("Stroke width: ", 1, 50, 20)
    else:
        genModel = st.selectbox(
            "Generator-Model:", options=['DCGAN', 'CWGAN-GP'])
        if genModel == 'CWGAN-GP':
            number = st.slider("CWGAN-GP Label: ", 0, 9, 0)
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
    if genModel == 'CWGAN-GP':
        generate = st.button(f"Generate {number} using {genModel}")
    else:
        generate = st.button(f"Generate digit using {genModel}")
    if 'fake' not in st.session_state or generate:
        noise = torch.randn(1, 100, 1, 1)
        if genModel == 'CWGAN-GP':
            fake = cgan(noise, torch.tensor([number]))
        else:
            fake = dcgan(noise)
        fake = fake.reshape(28, 28, 1).detach().numpy().squeeze()
        fake = (fake - np.min(fake))/np.ptp(fake)
        fake = Image.fromarray(fake)
        fake = fake.resize((280, 280), resample=Image.NEAREST)
        st.session_state['fake'] = np.array(fake)
    st.image(st.session_state['fake'], width=280)

result = st.button(f"Predict with {framework}")

if result:
    if page == "Draw":
        image = Image.fromarray((canvas.image_data[:, :, 0]).astype(np.uint8))
    else:
        image = Image.fromarray((st.session_state['fake']*255).astype(np.uint8))
    image = image.resize((28, 28))

    if framework == 'Pytorch':
        DataTransform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,))])
        tensor = DataTransform(image)
        image_tensor = tensor.unsqueeze_(0)
        with torch.no_grad():
            outputs = torch.exp(PytorchModel(image_tensor))
            outputs = outputs.squeeze().detach().numpy() ** 0.2
            ind_max = np.where(outputs == max(outputs))[0][0]

    elif framework == 'Keras':
        array = np.array(image)
        array = array.reshape(1, 28, 28, 1)
        array = (array - np.mean(array))/np.std(array)
        outputs = KerasModel.predict(array).squeeze() ** 0.2
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
    st.markdown('<img src="https://raw.githubusercontent.com/foersterrobert/MNIST-Drawer/master/GANs/assets/gan.gif" width="80%" style="display: block; margin: auto;"/>', unsafe_allow_html=True)
    st.markdown("---")
    st.subheader(f'{genModel} Generator Architecture')
    if genModel == 'DCGAN':
        st.code(
        """
class DCGAN(nn.Module):
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

    else:
        st.code(
        """
class CGAN(nn.Module):
    def __init__(self, channels_noise, channels_img, img_size, num_classes, embed_size):
        super().__init__()
        # Input: N x channels_noise | 1 x 100
        self.gen = nn.Sequential(
            self._block(channels_noise+embed_size, 512, 4, 1, 0),  # img: 4x4x432
            self._block(512, 512, 3, 1, 1),  # img: 4x4x512
            self._block(512, 448, 4, 1, 0),  # img: 7x7x512
            self._block(448, 448, 3, 1, 1),  # img: 7x7x512
            self._block(448, 256, 4, 2, 1),  # img: 14x14x448
            self._block(256, 256, 3, 1, 1),  # img: 14x14x448
            self._block(256, 128, 4, 2, 1),  # img: 28x28x112
            nn.ConvTranspose2d(
                128, channels_img, kernel_size=3, stride=1, padding=1
            ),
            # Output: N x channels_img | 28x28x1
            nn.Tanh(),
        )
        self.embed = nn.Embedding(num_classes, embed_size)

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

    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, embedding], dim=1)
        return self.gen(x)
        """
        )

st.markdown("---")
if framework == 'Pytorch':
    st.subheader(
        f'Model: {framework} | Test-Accuracy: 99.61% in Kaggle')
    st.write(
        'Data-Augmentation: Rotation & Translation, Batchsize: 64, Epochs: 100, Learning-Rate: 0.001, Optimizer: Adam')
    st.code(
        '''
class PytorchDrawer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            self._block(1, 32, 3),
            self._block(32, 48, 3),
            self._block(48, 64, 3),
            self._block(64, 80, 3),
            self._block(80, 96, 3),
            self._block(96, 112, 3),
            self._block(112, 128, 3),
            self._block(128, 144, 3),
            self._block(144, 160, 3),
            self._block(160, 176, 3),
            Flatten(),
            nn.Linear(11264, 10, bias=False),
            nn.BatchNorm1d(10),
        )

    def _block(self, input_dim, output_dim, kernel_size):
        return nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.seq(x)
        return F.log_softmax(x, dim=1)

class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x.permute(0, 2, 3, 1), 1)
        '''
    )

elif framework == 'Keras':
    st.subheader(
        f'Model: {framework} | Test-Accuracy: 99.44%')
    st.write(
        'Batchsize: 64, Epochs: 30')
    st.code(
        '''
model = keras.Sequential(
    [
        # 1x28x28
        Conv2D(filters = 32, kernel_size = 5, strides = 1, padding="same", activation = 'relu', input_shape = (28,28,1), kernel_regularizer=l2(0.0005)),
        # 32x28x28
        Conv2D(filters = 32, kernel_size = 5, strides = 1, padding="same", use_bias=False),
        # 32x28x28
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size = 2, strides = 2),
        Dropout(0.25),
        # 32x14x14
        Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = 'relu', kernel_regularizer=l2(0.0005)),
        # 64x12x12
        Conv2D(filters = 64, kernel_size = 3, strides = 1, use_bias=False),
        # 64x10x10
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size = 2, strides = 2), # 64x5x5
        Dropout(0.25),
        Flatten(),
        Dense(units = 256, use_bias=False), # 256
        BatchNormalization(),
        Activation('relu'),
        Dense(units = 128, use_bias=False), # 128
        BatchNormalization(),
        Activation('relu'),
        Dense(units = 84, use_bias=False), # 84
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.25),
        Dense(units = 10, activation = 'softmax') # 10
    ]
)
        '''
    )

elif framework == 'scikit-learn':
    st.subheader(
        f'Model: {framework} | Test-Accuracy: 96.78%')
    st.write('Support Vector Classification')
    st.code(
        '''
clf = svm.SVC(gamma=0.001, probability=True)
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