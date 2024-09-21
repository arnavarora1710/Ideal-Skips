import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleDict()  # Store layers in a dictionary
        self.skip_connections = {}     # Store skip connections between layers
        self.input_size = input_size
        self.output_size = output_size

    def add_layer(self, name, layer):
        """
        Adds a new layer to the network.
        :param name: The name of the layer (string)
        :param layer: The layer object (e.g., nn.Linear, nn.ReLU)
        """
        self.layers[name] = layer

    def add_skip_connection(self, from_layer, to_layer):
        """
        Adds a skip connection between two specified layers.
        :param from_layer: The name of the source layer
        :param to_layer: The name of the target layer
        """
        self.skip_connections[(from_layer, to_layer)] = True

    def remove_skip_connection(self, from_layer, to_layer):
        """
        Removes a skip connection between two layers.
        :param from_layer: The name of the source layer
        :param to_layer: The name of the target layer
        """
        if (from_layer, to_layer) in self.skip_connections:
            del self.skip_connections[(from_layer, to_layer)]

    def apply_skip(self, x, from_layer, to_layer):
        """
        Applies the skip connection between specified layers if it exists.
        :param x: The input tensor
        :param from_layer: The source layer name
        :param to_layer: The target layer name
        :return: The output tensor after applying the skip connection
        """
        if (from_layer, to_layer) in self.skip_connections:
            return x + self.layers[to_layer](x)
        return x

    def forward(self, x):
        """
        Forward pass through the network with skip connections.
        :param x: Input tensor
        :return: Output tensor
        """
        outputs = {}
        for name, layer in self.layers.items():
            if name == 'input':
                outputs[name] = layer(x)
            else:
                previous_layer = list(self.layers.keys())[list(self.layers.keys()).index(name) - 1]
                x = outputs[previous_layer]
                outputs[name] = self.apply_skip(x, previous_layer, name)

        return outputs[list(self.layers.keys())[-1]]

    def train_model(self, train_loader, criterion, optimizer, num_epochs=10):
        """
        Trains the model on the given data.
        :param train_loader: DataLoader with training data
        :param criterion: Loss function
        :param optimizer: Optimizer function
        :param num_epochs: Number of training epochs
        """
        self.train()
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    def evaluate_model(self, test_loader, criterion):
        """
        Evaluates the model on the given test data.
        :param test_loader: DataLoader with test data
        :param criterion: Loss function
        :return: Test loss
        """
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        print(f"Test Loss: {total_loss / len(test_loader)}")
        return total_loss / len(test_loader)

    def get_layer_weights(self, layer_name):
        """
        Returns the weights of a specified layer.
        :param layer_name: The name of the layer whose weights are to be retrieved
        :return: Weights of the layer as a tensor, or None if the layer has no weights
        """
        if layer_name in self.layers:
            layer = self.layers[layer_name]
            if hasattr(layer, 'weight'):
                return layer.weight.data
        # print(f"Layer '{layer_name}' does not have weights or does not exist.")
        return None

    def create_model(self, n_layers):
        self.add_layer('input', nn.Linear(self.input_size, 64))

        # hidden layers
        for i in range(1, n_layers + 1):
            self.add_layer(f'relu{i}', nn.ReLU())
            self.add_layer(f'linear{i}', nn.Linear(64, 64))

        self.add_layer('output', nn.Linear(64, self.output_size))