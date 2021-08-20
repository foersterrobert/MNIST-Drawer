# Importing necessary libraries
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os

kwargs = {} # cuda gpu?

train_data = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, 
                        transform=transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))])),
                        batch_size=128, shuffle=True, **kwargs)

class PytorchDrawer(nn.Module):
    def __init__(self):
        super(PytorchDrawer, self).__init__()
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

model = PytorchDrawer()


optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)
def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_data):
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.nll_loss
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_id * len(data), len(train_data.dataset),
                    100. * batch_id / len(train_data), loss.data))


for epoch in range(1, 20):
    train(epoch)

model.eval()


model_folder_path = './model'
file_name='Pytorch.pth'
if not os.path.exists(model_folder_path):
    os.makedirs(model_folder_path)

file_name = os.path.join(model_folder_path, file_name)
torch.save(model, file_name)