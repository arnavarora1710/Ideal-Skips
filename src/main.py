import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from neural_net import NeuralNet

# Define the NeuralNet class (already defined in the previous response)

# Define the model architecture with 10 layers
input_size = 28 * 28  # MNIST images are 28x28 pixels
output_size = 10      # 10 classes for digits 0-9
model = NeuralNet(input_size, output_size)

# Adding 10 layers to the network
model.add_layer('input', nn.Linear(input_size, 64))   # Layer 1
model.add_layer('relu1', nn.ReLU())                  # Layer 2
model.add_layer('linear2', nn.Linear(64, 128))       # Layer 3
model.add_layer('relu2', nn.ReLU())                  # Layer 4
model.add_layer('linear3', nn.Linear(128, 256))      # Layer 5
model.add_layer('relu3', nn.ReLU())                  # Layer 6
model.add_layer('linear4', nn.Linear(256, 128))      # Layer 7
model.add_layer('relu4', nn.ReLU())                  # Layer 8
model.add_layer('linear5', nn.Linear(128, 64))       # Layer 9
model.add_layer('output', nn.Linear(64, output_size)) # Layer 10

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
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the images
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
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the images
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    print(f"Test Loss: {total_loss / len(test_loader)}, Accuracy: {accuracy * 100:.2f}%")

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=5)

# Evaluate the model
evaluate_model(model, test_loader, criterion)