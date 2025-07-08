# 225150200111004_1 HAIKAL THORIQ ATHAYA_1
# 225150200111008_2 MUHAMMAD ARSYA ZAIN YASHIFA_2
# 225150201111001_3 JAVIER AAHMES REANSYAH_3
# 225150201111003_4 MUHAMMAD HERDI ADAM_4

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(ROOT_DIR, 'Data')
    MODEL_DIR = os.path.join(ROOT_DIR, 'Model')

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.2f}%")

    example_input = torch.randn(1, 1, 28, 28)
    scripted_model = torch.jit.trace(model, example_input)
    model_path = os.path.join(MODEL_DIR, 'mnist_cnn.pt')
    scripted_model.save(model_path)
    print(f"TorchScript model saved to {model_path}")