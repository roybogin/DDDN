import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1024)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        print(x, x.shape)
        logits = torch.max(x, dim=1)
        return logits