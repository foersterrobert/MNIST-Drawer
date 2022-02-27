import torch.nn as nn
import torch
import torch.nn.functional as F

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