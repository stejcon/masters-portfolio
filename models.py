import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class BaseResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        self.num_classes = num_classes

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return (0, x)


class StrippedResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        self.num_classes = num_classes
        self.exit5 = torch.nn.LazyLinear(self.num_classes)
        self.exit6 = torch.nn.LazyLinear(self.num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit5(y)
        with torch.no_grad():
            pk = F.softmax(y, dim=1)
            entr = -torch.sum(pk * torch.log(pk + 1e-20))
            if torch.all(entr < 0.51512845993042):
                return (5, y)
        x = self.layer3(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit6(y)
        with torch.no_grad():
            pk = F.softmax(y, dim=1)
            entr = -torch.sum(pk * torch.log(pk + 1e-20))
            if torch.all(entr < 300000):
                return (6, y)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        self.num_classes = num_classes
        self.exit1 = torch.nn.LazyLinear(self.num_classes)
        self.exit2 = torch.nn.LazyLinear(self.num_classes)
        self.exit3 = torch.nn.LazyLinear(self.num_classes)
        self.exit4 = torch.nn.LazyLinear(self.num_classes)
        self.exit5 = torch.nn.LazyLinear(self.num_classes)
        self.exit6 = torch.nn.LazyLinear(self.num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit1(y)
        with torch.no_grad():
            pk = F.softmax(y, dim=1)
            entr = -torch.sum(pk * torch.log(pk + 1e-20))
            if torch.all(entr < 0.0):
                return (1, y)
        x = self.maxpool(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit2(y)
        with torch.no_grad():
            pk = F.softmax(y, dim=1)
            entr = -torch.sum(pk * torch.log(pk + 1e-20))
            if torch.all(entr < 2.331410984290952e-30):
                return (2, y)
        x = self.layer0(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit3(y)
        with torch.no_grad():
            pk = F.softmax(y, dim=1)
            entr = -torch.sum(pk * torch.log(pk + 1e-20))
            if torch.all(entr < 0.0):
                return (3, y)
        x = self.layer1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit4(y)
        with torch.no_grad():
            pk = F.softmax(y, dim=1)
            entr = -torch.sum(pk * torch.log(pk + 1e-20))
            if torch.all(entr < 1.5324721572728675e-24):
                return (4, y)
        x = self.layer2(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit5(y)
        with torch.no_grad():
            pk = F.softmax(y, dim=1)
            entr = -torch.sum(pk * torch.log(pk + 1e-20))
            if torch.all(entr < 0.51512845993042):
                return (5, y)
        x = self.layer3(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit6(y)
        with torch.no_grad():
            pk = F.softmax(y, dim=1)
            entr = -torch.sum(pk * torch.log(pk + 1e-20))
            if torch.all(entr < 300000):
                return (6, y)


class HalfResNet(ResNet):
    def __init__(self, block, layers, num_classes=10):
        super(HalfResNet, self).__init__(block, layers, num_classes=10)
        self.linear = nn.Linear(61952, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return (0, x)


class BranchedResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(BranchedResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        self.num_classes = num_classes

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = nn.Linear(y.size(1), self.num_classes)(y)
        y = torch.nn.functional.softmax(y, dim=1)
        entropy = -torch.sum(y * torch.log2(y + 1e-20), dim=1)
        if torch.all(entropy < 300):
            return (1, y)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return (0, x)
