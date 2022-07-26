
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets


BATCH_SIZE = 256
EPOCHS = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                       torchvision.transforms.RandomRotation((-10, 10)),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=BATCH_SIZE, shuffle=True)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(in_channels=1,
                                             out_channels=32,
                                             kernel_size=5,
                                             stride=1, padding=0))
        layer1.add_module('relu1', nn.ReLU())
        layer1.add_module('BN1', nn.BatchNorm2d(32))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(in_channels=32, out_channels=32,
                                             kernel_size=5, stride=1, padding=0))
        layer2.add_module('relu2', nn.ReLU())
        layer2.add_module('BN2', nn.BatchNorm2d(32))
        layer2.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        layer2.add_module('dropout2', nn.Dropout(0.25))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv3', nn.Conv2d(in_channels=32, out_channels=64,
                                             kernel_size=3, stride=1, padding=0))
        layer3.add_module('BN3', nn.BatchNorm2d(64))
        layer3.add_module('relu3', nn.ReLU())
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('conv4', nn.Conv2d(in_channels=64, out_channels=64,
                                             kernel_size=3, stride=1, padding=0))
        layer4.add_module('BN4', nn.BatchNorm2d(64))
        layer4.add_module('relu4', nn.ReLU())
        layer4.add_module('pool4', nn.MaxPool2d(kernel_size=2, stride=2))
        layer4.add_module('dropout4', nn.Dropout(0.25))
        self.layer4 = layer4

        layer5 = nn.Sequential()
        layer5.add_module('fc1',nn.Linear(576,256))
        layer5.add_module('fc_relu1',nn.ReLU())
        layer5.add_module('fc2',nn.Linear(256,64))
        layer5.add_module('fc_relu2',nn.ReLU())
        layer5.add_module('fc3',nn.Linear(64,10))

        self.layer5 = layer5

    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        conv4 = self.layer4(conv3)
        fc_input = conv4.view(conv4.size(0), -1)
        fc_out = self.layer5(fc_input)
        return F.log_softmax(fc_out,dim=1)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


model = ConvNet().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.001
                      , momentum=0.5)
model.apply(weight_init)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


#
#
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  
            pred = output.max(1, keepdim=True)[1]  
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    torch.save(model.state_dict(), './model.pth')
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, EPOCHS + 1):
    model.load_state_dict(torch.load('model.pth'))
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)
