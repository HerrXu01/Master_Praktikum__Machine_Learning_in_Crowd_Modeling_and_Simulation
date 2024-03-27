import torch
import torch.nn as nn

class PedNeuralNetwork(nn.Module):
    #def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        layers = []
        layers.append(nn.Sequential(nn.Linear(input_size,hidden_sizes[0]),nn.BatchNorm1d(hidden_sizes[0]), nn.ReLU()))
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Sequential(nn.Linear(hidden_sizes[i-1],hidden_sizes[i]),nn.BatchNorm1d(hidden_sizes[i]),nn.ReLU()))
       
        layers.append(nn.Linear(hidden_sizes[-1],output_size))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
