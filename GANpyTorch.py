import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super().__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img | N = W x H | 28 x 28 x 1
            nn.Conv2d(
                channels_img, features_d * 2, kernel_size=4, stride=2, padding=1
            ), # 14x14x56
            nn.LeakyReLU(0.2),
            self._block(features_d * 2, features_d * 4, 3, 1, 1), # 14x14x112
            self._block(features_d * 4, features_d * 8, 3, 1, 1), # 14x14x224
            self._block(features_d * 8, features_d * 16, 4, 2, 1), # 7x7x448
            # After all _block img output is 7x7x448 (Conv2d below makes into 1x1x1 -> 1)
            nn.Conv2d(features_d * 16, 1, kernel_size=7, stride=2, padding=0),
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

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

if __name__ == "__main__":
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 128
    IMAGE_SIZE = 28
    CHANNELS_IMG = 1
    CHANNELS_NOISE = 100
    NUM_EPOCHS = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_data = DataLoader(datasets.MNIST('data', train=True, download=True, 
                transform=transforms.Compose([
                        transforms.Resize(IMAGE_SIZE), 
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))])
                ),
                batch_size=BATCH_SIZE, shuffle=True)
    gen = Generator(CHANNELS_NOISE, CHANNELS_IMG, IMAGE_SIZE).to(device)
    disc = Discriminator(CHANNELS_IMG, IMAGE_SIZE).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, CHANNELS_NOISE, 1, 1).to(device) # 32 digits for Tensorboard
    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")
    step = 0
    for epoch in range(NUM_EPOCHS):
        gen_loss = 0.0
        disc_loss = 0.0
        for batch_idx, (real, _) in enumerate(train_data):
            real = real.to(device)

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            noise = torch.randn(BATCH_SIZE, CHANNELS_NOISE, 1, 1).to(device)
            fake = gen(noise)
            disc_real = disc(real).reshape(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc_loss += lossD
            disc.zero_grad()
            lossD.backward()
            opt_disc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            # where the second option of maximizing doesn't suffer from
            # saturating gradients
            output = disc(fake).reshape(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen_loss += lossG
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

                    writer_real.add_scalar('Loss Discriminator',
                                disc_loss / 100,
                                epoch * len(train_data) + batch_idx)
                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    
                    writer_fake.add_scalar('Loss Generator',
                                gen_loss / 100,
                                epoch * len(train_data) + batch_idx)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1
                gen_loss = 0
                disc_loss = 0


    torch.save(gen.state_dict(), "model/PytorchGAN.pth")
