import torch.nn as nn
import torch

class PytorchDrawer(nn.Module):
    def __init__(self):
        super().__init__()
        # 1x28x28
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2), # 32x28x28
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, 1, 2, bias=False), # 32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x14x14
            nn.Conv2d(32, 64, 3, 1), # 64x12x12
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, bias=False), # 64x10x10
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64x5x5
            Flatten(),
            nn.Linear(64*5*5, 256, bias=False), # 256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128, bias=False), # 128
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 84, bias=False), # 84
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(84, 10), # 10
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn(x)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

        
class DCGAN(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super().__init__()
        self.gen = nn.Sequential(
            # Input: N x channels_noise | 1 x 100
            self._block(channels_noise, features_g * 32, 7, 1, 0),  # img: 7x7x896
            self._block(features_g * 32, features_g * 16, 4, 2, 1),  # img: 14x14x448
            self._block(features_g * 16, features_g * 8, 3, 1, 1),  # img: 14x14x224
            self._block(features_g * 8, features_g * 4, 3, 1, 1),  # img: 14x14x112
            nn.ConvTranspose2d(
                features_g * 4, channels_img, kernel_size=4, stride=2, padding=1
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