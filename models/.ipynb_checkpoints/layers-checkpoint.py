import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=heads, concat=True)
        self.conv2 = GATConv(out_channels * heads, out_channels, heads=heads, concat=False)

    def forward(self, x, edge_index, edge_weights):
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_weights))
        x = self.conv2(x, edge_index, edge_attr=edge_weights)
        return x
    
    
class CrossAttention(nn.Module):
    def __init__(self, D):
        super(CrossAttention, self).__init__()
        self.W_q = nn.Parameter(torch.randn(D, D))
        self.D = D
    
    def forward(self, x):  # x (B, L, D)
        Q, K, V = torch.matmul(x, self.W_q), x, x,
        attention_scores = torch.matmul(Q, K.transpose(1, 2)) / (2 * self.D) ** 0.5  # (B, L, L)      
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, L, L)
        output = torch.matmul(attention_weights, V)  # (B, L, L) * (B, L, D) -> (B, L, D)
        return output
