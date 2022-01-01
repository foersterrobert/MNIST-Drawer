import torch.nn as nn
import torch

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
        self.img_size = img_size
        self.gen = nn.Sequential(
            # Input: N x channels_noise | 1 x 100
            self._block(channels_noise+embed_size, img_size * 32, 7, 1, 0),  # img: 7x7x896
            self._block(img_size * 32, img_size * 16, 4, 2, 1),  # img: 14x14x448
            self._block(img_size * 16, img_size * 8, 3, 1, 1),  # img: 14x14x224
            self._block(img_size * 8, img_size * 4, 3, 1, 1),  # img: 14x14x112
            nn.ConvTranspose2d(
                img_size * 4, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img | 28x28x1
            nn.Tanh(), # outputs values between -1 and 1
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