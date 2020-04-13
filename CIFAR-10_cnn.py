import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import numpy as np

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
# Hyper Params
n_epochs = 100
batch_size = 100

def lr_sced(cur_epoch):
    if cur_epoch < 50:
        lr = 0.001
    elif cur_epoch < 80:
        lr = 0.0005
    else:
        lr = 0.0003
    return lr
        


# dataset has PILImage images of range [0, 1]
# We transform them to Tensors of normilised range [-1, 1]

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Convolutional Nueral Net
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv_bn1 = nn.BatchNorm2d(32)
        self.conv_bn2 = nn.BatchNorm2d(64)

        self.drop = nn.Dropout(0.25)

        self.fc1 = nn.Linear(64*5*5, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.out = nn.Linear(500, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv_bn1(x)
        x = self.pool(x)
        x = self.drop(x)

        x = F.relu(self.conv2(x))
        x = self.conv_bn2(x)
        x = self.pool(x)
        x = self.drop(x)

        x = x.reshape(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.out(x)
        
        return x

# creating model
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
total_steps = len(train_loader)
for epoch in range(n_epochs):
    lr = lr_sced(epoch)
    print(f'Epoch{epoch} -- Lr: {lr}')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i, (images, lables) in enumerate(train_loader):
        images = images.to(device)
        lables = lables.to(device)

        outputs = model(images)
        loss = criterion(outputs, lables)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch {epoch+1} / {n_epochs}, Step {i+1} / {total_steps}, Loss {loss.item():.4f}')

print('Training Finished')

# testing loop
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels.to(device)).sum().item()


    acc = 100 * n_correct / n_samples
print(f'Accuracy of the network is {acc}%')
