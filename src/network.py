import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import print_d, Level


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
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


class GoogLeNet:
    def __init__(self):
        print_d("Loading model", Level.INFO)
        self.model =  torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=False, init_weights=False)
        print_d("Model loaded", Level.INFO)

    def forward(self, x):
        output = self.model.forward(x)
        return F.log_softmax(output[0], dim=1)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def parameters(self):
        return self.model.parameters()

    def __call__(self, x):
        return self.forward(x)

    def __getattr__(self, attr):
        return getattr(self.model,attr)
