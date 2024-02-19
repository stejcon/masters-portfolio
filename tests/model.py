import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.max_pool2d(F.relu(nn.Conv2d(1, 6, 5)(x)), (2, 2))
        x = F.max_pool2d(F.relu(nn.Conv2d(6, 16, 5)(x)), 2)
        print(f'{3}')
        x = torch.flatten(x, 1)
        x = F.relu(nn.Linear(16 * 5 * 5, 120)(x))
        x = F.relu(nn.Linear(120, 84)(x))
        x = nn.Linear(84, 10)(x)
        return x