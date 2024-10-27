import sys
sys.path.append(r'/mnt/c/Users/jahuz/Links/BP/_annotation')

from paths import *


class BackboneCNN(nn.Module):
    def __init__(self):
        super(BackboneCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        p1 = self.pool(c1)
        c2 = F.relu(self.conv2(p1))
        p2 = self.pool(c2)
        c3 = F.relu(self.conv3(p2))
        p3 = self.pool(c3)
        c4 = F.relu(self.conv4(p3))
        p4 = self.pool(c4)
        return p1, p2, p3, p4

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = models.resnet18(pretrained=True)
        
        # Extract layers from ResNet
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # First block
        self.layer2 = resnet.layer2  # Second block
        self.layer3 = resnet.layer3  # Third block
        self.layer4 = resnet.layer4  # Fourth block

    def forward(self, x):
        # Pass through ResNet layers and capture feature maps
        p1 = self.relu(self.bn1(self.conv1(x)))  # Initial conv/bn/relu
        p1 = self.maxpool(p1)
        
        p2 = self.layer1(p1)  # First block output
        p3 = self.layer2(p2)  # Second block output
        p4 = self.layer3(p3)  # Third block output
        p5 = self.layer4(p4)  # Fourth block output
        
        return p2, p3, p4, p5  # Return the feature maps