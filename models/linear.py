'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch 

class FNN3(nn.Module):
    def __init__(self):
        super(FNN3, self).__init__()
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class FNN1(nn.Module):
    def __init__(self):
        super(FNN1, self).__init__()
        self.fc = nn.Linear(3072, 10)

    def forward(self, x):
        out = torch.flatten(x, 1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    model = FNN1()
    print(model(torch.rand(10, 3, 32, 32)))