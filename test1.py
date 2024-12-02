import torch
import torch.nn as nn
import torch.nn.functional as F


import pandas as pd

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        # x = (x-x.mean())/x.std()
        return x
mlp = MLP(5, 4, 5)
x = torch.randn([3, 4, 5])
print(mlp(x).shape)