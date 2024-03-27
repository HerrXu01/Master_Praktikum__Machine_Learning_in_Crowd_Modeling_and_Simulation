import torch
import torch.nn as nn

class NN(nn.Module):
    """
    A simple feedforward neural network using ReLU activations, built with PyTorch.

    The network is constructed dynamically based on the specified number of layers and nodes per layer.
    The input layer expects a fixed size of 21 features, and the output layer consists of a single neuron.

    Parameters:
    - num_layers (int): The total number of hidden layers plus the output layer.
    - num_nodes (list): A list specifying the number of nodes in each hidden layer.

    The final layer does not use an activation function, making this network suitable for regression tasks.
    """
    def __init__(self, num_layers: int, num_nodes: list):
        super(NN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(21, num_nodes[0]))
        
        for i in range(1, num_layers):
            self.layers.append(nn.Linear(num_nodes[i-1], num_nodes[i]))

        self.layers.append(nn.Linear(num_nodes[-1], 1))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x