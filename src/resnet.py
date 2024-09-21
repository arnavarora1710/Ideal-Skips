import torch
import torch.nn as nn

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        # Define any general layer
        self.input_layer = None
        self.hidden_layers = []
        self.output_layer = nn.Linear(20, 1)

        # Dictionary to hold skip connections
        self.skip_connections = {}
    
    def add_layer(self, new_layer):
        self.hidden_layers.append(new_layer)

    def add_skip_connection(self, from_layer, to_layer):
        """
        Adds a skip connection from one layer to another.
        :param from_layer: Name of the source layer (e.g., 'layer1')
        :param to_layer: Name of the target layer (e.g., 'layer3')
        """
        self.skip_connections[(from_layer, to_layer)] = True

    def apply_skip(self, x, from_layer, to_layer):
        """
        Applies skip connection between layers if defined.
        :param x: Input tensor from the source layer
        :param from_layer: Source layer name
        :param to_layer: Target layer name
        :return: Modified tensor if skip connection exists, otherwise original tensor
        """
        if (from_layer, to_layer) in self.skip_connections:
            return x + getattr(self, to_layer)  # Example: Adding the skip output
        return x

    def forward(self, x):
        # random example
        x1 = self.layer(x)  # Pass through first layer
        out = self.output_layer(x1)  # Output layer
        return out