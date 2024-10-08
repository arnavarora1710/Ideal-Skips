# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# numpy
import numpy as np

# custom
from neural_net import NeuralNet
from learning_array import gen_learning_array
from matrix import computeMatrix
from random_walk import random_walk_with_updates

# Global Vars
LAYERS = 100
STEPS = 5
EPOCHS_INIT = 0
EPOCHS_TRAIN = 5

input_size = 28 * 28 
output_size = 10
model = NeuralNet(input_size=input_size, output_size=output_size)
model.create_model(n_layers=LAYERS)

# Define transformations for the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST dataset
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Define DataLoader for training and testing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function within the NeuralNet class
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.view(inputs.size(0), -1) # Flatten the images
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

# Evaluation function within the NeuralNet class
def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.view(inputs.size(0), -1) # Flatten the images
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    print(f"Test Loss: {total_loss / len(test_loader)}, Accuracy: {accuracy * 100:.2f}%")

def find_optimal_configurations():
    train_model(model, train_loader, criterion, optimizer, num_epochs=EPOCHS_INIT)

    model_weights = []
    for layer_name in model.layers.keys():
        weights = model.get_layer_weights(layer_name)
        if weights is not None:
            model_weights.append(weights)
    model_weights.pop()
    model_weights.pop(0)

    L = gen_learning_array(model_weights)

    graph = computeMatrix(L)

    choicer = [x for x in range(1, LAYERS - 1)]
    start = np.random.choice(choicer)
    result = random_walk_with_updates(LAYERS, start, graph, STEPS)
    for stuff in result:
        print(stuff)
        model.add_skip_connection(stuff[0], stuff[1])
    train_model(model, train_loader, criterion, optimizer, num_epochs=EPOCHS_TRAIN)

find_optimal_configurations()
evaluate_model(model, test_loader, criterion)