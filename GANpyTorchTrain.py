import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

sample_noise = torch.tensor([[
         [[ 1.6720]],

         [[-0.9494]],

         [[ 0.1965]],

         [[-0.7145]],

         [[ 0.3630]],

         [[-0.4528]],

         [[-1.0217]],

         [[-0.3492]],

         [[ 0.6904]],

         [[ 0.1228]],

         [[ 1.2227]],

         [[ 0.8850]],

         [[-0.6011]],

         [[ 0.1207]],

         [[-0.0731]],

         [[-0.8952]],

         [[ 0.1624]],

         [[-1.3364]],

         [[ 2.0384]],

         [[ 1.1124]],

         [[-0.4746]],

         [[-1.3475]],

         [[-0.0547]],

         [[ 0.0763]],

         [[-0.1194]],

         [[-1.7790]],

         [[ 1.1536]],

         [[-0.0669]],

         [[-1.2572]],

         [[-0.8158]],

         [[-0.3454]],

         [[-0.4257]],

         [[-0.6719]],

         [[-0.0427]],

         [[-0.4244]],

         [[ 0.1909]],

         [[-0.1003]],

         [[ 0.0893]],

         [[ 2.2683]],

         [[ 0.2261]],

         [[ 0.1692]],

         [[ 1.8331]],

         [[-0.6978]],

         [[-2.4364]],

         [[-1.2574]],

         [[-0.1052]],

         [[ 1.5563]],

         [[-0.3468]],

         [[-1.9549]],

         [[-1.3695]],

         [[-1.3819]],

         [[-0.4200]],

         [[ 0.6477]],

         [[-0.6517]],

         [[ 0.0629]],

         [[ 0.4844]],

         [[ 1.3450]],

         [[ 0.2447]],

         [[ 0.6959]],

         [[ 1.7464]],

         [[ 1.9081]],

         [[-0.6873]],

         [[ 0.3319]],

         [[-0.6513]],

         [[ 0.7896]],

         [[ 0.6763]],

         [[-0.8106]],

         [[-0.0834]],

         [[-0.2298]],

         [[-0.3516]],

         [[-0.5408]],

         [[ 0.5488]],

         [[ 0.4009]],

         [[-1.6072]],

         [[ 0.1341]],

         [[-1.4107]],

         [[-0.7686]],

         [[ 1.2790]],

         [[-0.7350]],

         [[-1.0894]],

         [[-0.5234]],

         [[ 0.3022]],

         [[ 0.8272]],

         [[ 0.0075]],

         [[ 0.3540]],

         [[ 0.0159]],

         [[-1.6494]],

         [[-0.7925]],

         [[-0.1387]],

         [[-0.2411]],

         [[ 1.1053]],

         [[ 1.2355]],

         [[ 0.6732]],

         [[ 1.3893]],

         [[ 0.9330]],

         [[ 0.1366]],

         [[-0.7038]],

         [[-0.3808]],

         [[-1.9904]],

         [[-0.3271]]]])


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
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
        return self.net(x)

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

if __name__ == "__main__":
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 128
    IMAGE_SIZE = 64
    CHANNELS_IMG = 1
    NOISE_DIM = 100
    NUM_EPOCHS = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_data = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, 
                transform=transforms.Compose([
                        transforms.Resize(IMAGE_SIZE), 
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))])
                ),
                batch_size=BATCH_SIZE, shuffle=True)
    gen = Generator(NOISE_DIM, CHANNELS_IMG, IMAGE_SIZE).to(device)
    disc = Discriminator(CHANNELS_IMG, IMAGE_SIZE).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device) # 32 digits for Tensorboard
    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")
    step = 0
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, _) in enumerate(train_data):
            real = real.to(device)

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
            fake = gen(noise)
            disc_real = disc(real).reshape(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward()
            opt_disc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            # where the second option of maximizing doesn't suffer from
            # saturating gradients
            output = disc(fake).reshape(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train_data)} \
                    Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True
                    )

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1


    torch.save(gen.state_dict(), "model/PytorchGAN.pth")
