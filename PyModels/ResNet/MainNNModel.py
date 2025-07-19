# ResNet Implemenation

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, channels, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        # Ensures that the input and output channels are the same dimension
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(self.expansion * channels)
            )

    def forward(self, x):
        inputTensor = x

        outputTensor = self.conv1(inputTensor)
        outputTensor = self.bn1(outputTensor)

        outputTensor = self.relu(outputTensor)

        outputTensor = self.conv2(outputTensor)
        outputTensor = self.bn2(outputTensor)

        

        # Residual Connection
        outputTensor += self.shortcut(inputTensor)

        outputTensor = self.relu(outputTensor)

        return outputTensor
    

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, channels, stride = 1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size = 1, stride = stride, bias = False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        # self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(self.expansion * channels)
            )

    def forward(self, x):
        inputTensor = x

        outputTensor = self.conv1(inputTensor)
        outputTensor = self.bn1(outputTensor)

        
        outputTensor = self.relu(outputTensor)

        outputTensor = self.conv2(outputTensor)
        outputTensor = self.bn2(outputTensor)

        outputTensor = self.relu(outputTensor)

        outputTensor = self.conv3(outputTensor)
        outputTensor = self.bn3(outputTensor)


        outputTensor += inputTensor

        outputTensor = self.relu(outputTensor)

        return outputTensor
    

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias= False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, channels, blocks, stride = 1):
        layers = []
        strides = [stride] + [1] * (blocks - 1) # Creates a list of strides for the layers, the first layer has the stride, the rest have 1

        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels * block.expansion # Updates the input channels for the next layer

        return nn.Sequential(*layers)

    
    def forward(self, x):
        inputTensor = x

        outputTensor = self.conv1(inputTensor)
        outputTensor = self.bn1(outputTensor)

        outputTensor = self.relu(outputTensor)

        outputTensor = self.maxpool(outputTensor)

        outputTensor = self.layer1(outputTensor)
        outputTensor = self.layer2(outputTensor)
        outputTensor = self.layer3(outputTensor)
        outputTensor = self.layer4(outputTensor)

        outputTensor = self.avgpool(outputTensor)
        outputTensor = torch.flatten(outputTensor, 1)
        outputTensor = self.fc(outputTensor)

        return outputTensor
    

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])