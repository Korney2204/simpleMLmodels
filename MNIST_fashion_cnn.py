import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms



data_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor())
data_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(data_train, batch_size=3, shuffle=True)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=3, shuffle=False)

# lables = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=12 , out_channels=24, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_bn1 = nn.BatchNorm2d(12)
        self.conv_bn2 = nn.BatchNorm2d(24)

        self.fc1 = nn.Linear(24*4*4, 200)
        self.out = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv_bn1(x)
        x = self.pool(x)
        

        x = F.relu(self.conv2(x))
        x = self.conv_bn2(x)
        x = self.pool(x)

        x = x.reshape(-1, 24*4*4)

        x = F.relu(self.fc1(x))

        x = self.out(x)

        return x


n_epochs = 5
learning_rate = 0.001
model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1000 == 0:
            print(f'Epoch {epoch+1},  Batch: {i+1}, Loss = {loss.item():.4f}')


with torch.no_grad():
    n_samples = 0
    n_correct = 0

    for images, labels in test_loader:

        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        n_correct += (predictions == labels).sum().item()
        n_samples += labels.size(0)

    
    acc = 100 * n_correct / n_samples

print(f'ACC {acc:.4f}')