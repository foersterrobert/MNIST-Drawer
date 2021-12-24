import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os

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

def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_data):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_id * len(data), len(train_data.dataset),
                    100. * batch_id / len(train_data), loss.data))

if __name__ == '__main__':
    train_data = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, 
                        transform=transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))])),
                        batch_size=64, shuffle=True)

    model = PytorchDrawer()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(1, 20):
        train(epoch)

    save = input('save? y ')
    if save == 'y':
        model.eval()
        model_folder_path = './model'
        file_name='Pytorch.pth'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(model, file_name)