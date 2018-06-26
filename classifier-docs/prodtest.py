import os
import torch
import torchvision.datasets
import torchvision.transforms as transforms
#trainset=torchvision.datasets.ImageFolder(root='home/aditya/research/dataset-dist/phase-01/training')

ROOT_DIR = os.getcwd()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
generic_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(), 
    transforms.Grayscale(),
    #transforms.CenterCrop(size=128),
    transforms.Resize((720,720)),
    transforms.Normalize((0.5), (2)),
    #transforms.ToTensor(),
])
"""
# trainset = torchvision.datasets.ImageFolder(root='/home/product.labs/dataset-dist/phase-01/training',transform=generic_transform)

"""
trainset = torchvision.datasets.ImageFolder(root='/home/product.labs/dataset-dist/phase-01/training',transform=transforms.Compose([transforms.ToPILImage(),transforms.Grayscale(),transforms.Resize((720,720)),transforms.ToTensor(),transforms.Normalize((0.5),(2))])
"""

trainset = torchvision.datasets.ImageFolder(root=ROOT_DIR+'classifier-docs/training',transform= transforms.Compose([transforms.ToTensor(),transforms.ToPILImage(),
transforms.transforms.Grayscale(num_output_channels = 1), transforms.Resize((720,720)),transforms.ToTensor()])) 

testset = torchvision.datasets.ImageFolder(root=ROOT_DIR+'classifier-docs/testing',transform= transforms.Compose([transforms.ToTensor(),transforms.ToPILImage(),
transforms.transforms.Grayscale(num_output_channels = 1), transforms.Resize((720,720)),transforms.ToTensor()]))

# batch size = 32 should have been 1

trainloader=torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=15)
testloader = torch.utils.data.DataLoader(testset, batach_size = 1, shuffle=True, num_workers= 15)

classes = ('fake','pristine')

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # layer1
        self.pool1 = nn.MaxPool2d(3,3)
        self.conv1 = nn.Conv2d(1,8,5,1)
        self.bn1 = nn.BatchNorm2d(8)
        self.apool1 = nn.AvgPool2d(kernel_size=5,stride=2)
        # layer 2
        self.conv2 = nn.Conv2d(8,16,kernel_size=3,stride=8)
        self.bn2 = nn.BatchNorm2d(16)
        self.apool2 = nn.AvgPool2d(kernel_size =5,stride=2)
        #layer 3
        self.conv3 = nn.Conv2d(16,32,kernel_size=3,stride= 16)
        self.bn3 = nn.BatchNorm2d(32)
        self.apool3 = nn.AvgPool2d(kernel_size=5,stride=2)
        #layer 4
        self.conv4 = nn.Conv2d(32,64,kernel_size=1,stride= 32)
        self.bn4 = nn.BatchNorm2d(64)
        self.apool4 = nn.AvgPool2d(kernel_size=5,stride=2)
        # layer 5
        self.conv5 = nn.Conv2d(64,128,kernel_size=1,stride= 64)
        self.bn5 = nn.BatchNorm2d(64)
        # self.GovalAvgPooling(tensor)
        #x = F.adaptive_avg_pool2d(x, (1, 1))
        #x = F.adaptive_avg_pool2d(x, (1, 1))
        self.fc1 = nn.Linear(128,84)
        self.fc2 = nn.Linear(84,2)

        """
        self.conv1 = nn.Conv2d(3, 6, 5,stride=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        """
    """
    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    """
    def forward(self, x):
        x = self.apool1(F.relu(self.bn1(self.conv1(self.pool1(x)))))
        x = self.apool2(F.relu(self.bn2(self.conv2(x))))
        #x = self.apool3(F.relu(self.bn3(self.conv3(x))))
        x = (F.relu((self.conv3(x))))
        #x = self.apool4(F.relu(self.bn4(self.conv4(x))))
        x = (F.relu(self.conv4(x)))
        #x = self.apool5(F.relu(self.bn5(self.conv5(x))))
        x = (F.relu(self.conv5(x)))      
        x = F.adaptive_avg_pool2d(x, (1, 1))
        #x = F.relu(self.fc1(x.transpose_(0,1)))
        x = x.view(x.size(0), -1)
        print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
     
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs,labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        #print(inputs)
        #print(labels)
        # forward + backward + optimize
        outputs = net(inputs)
        print(outputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i%10 == 0:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')

# testing
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
