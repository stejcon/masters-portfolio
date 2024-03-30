import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
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


class StrippedBiggerExitResNet(nn.Module):
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


class BiggerExitResNet(nn.Module):
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


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return (0, x)


class ResNet50Cifar10(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
        self.exit1 = torch.nn.LazyLinear(num_classes)
        self.exit2 = torch.nn.LazyLinear(num_classes)
        self.exit3 = torch.nn.LazyLinear(num_classes)
        self.exit4 = torch.nn.LazyLinear(num_classes)
        self.exit5 = torch.nn.LazyLinear(num_classes)
        self.exit6 = torch.nn.LazyLinear(num_classes)
        self.exit7 = torch.nn.LazyLinear(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit1(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.7168469168245792):
            return (1, y)
        x = self.bn1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit2(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.7035407423973083):
            return (2, y)
        x = self.relu(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit3(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.0647062361240387):
            return (3, y)
        x = self.maxpool(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit4(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.8355146628618241):
            return (4, y)
        x = self.layer1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit5(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.6622997450828553):
            return (5, y)
        x = self.layer2(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit6(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.8887244826555253):
            return (6, y)
        x = self.layer3(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit7(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.1911989336321132):
            return (7, y)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return (0, x)


class ResNet50Cifar100(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
        self.exit1 = torch.nn.LazyLinear(num_classes)
        self.exit2 = torch.nn.LazyLinear(num_classes)
        self.exit3 = torch.nn.LazyLinear(num_classes)
        self.exit4 = torch.nn.LazyLinear(num_classes)
        self.exit5 = torch.nn.LazyLinear(num_classes)
        self.exit6 = torch.nn.LazyLinear(num_classes)
        self.exit7 = torch.nn.LazyLinear(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit1(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.3460010796785356):
            return (1, y)
        x = self.bn1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit2(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.9012664079666137):
            return (2, y)
        x = self.relu(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit3(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 2.520589714050293):
            return (3, y)
        x = self.maxpool(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit4(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 2.4997169840335847):
            return (4, y)
        x = self.layer1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit5(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.0454509156942366):
            return (5, y)
        x = self.layer2(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit6(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.5064539313316345):
            return (6, y)
        x = self.layer3(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit7(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 2.96774370522704):
            return (7, y)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return (0, x)


class ResNet50QMNIST(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
        self.exit1 = torch.nn.LazyLinear(num_classes)
        self.exit2 = torch.nn.LazyLinear(num_classes)
        self.exit3 = torch.nn.LazyLinear(num_classes)
        self.exit4 = torch.nn.LazyLinear(num_classes)
        self.exit5 = torch.nn.LazyLinear(num_classes)
        self.exit6 = torch.nn.LazyLinear(num_classes)
        self.exit7 = torch.nn.LazyLinear(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit1(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.5841562747955322):
            return (1, y)
        x = self.bn1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit2(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.8885105848312378):
            return (2, y)
        x = self.relu(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit3(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.4497729420661927):
            return (3, y)
        x = self.maxpool(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit4(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.3224418622255325):
            return (4, y)
        x = self.layer1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit5(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.2198189795017242):
            return (5, y)
        x = self.layer2(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit6(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.0768353688716887):
            return (6, y)
        x = self.layer3(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit7(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.9760104127554223):
            return (7, y)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return (0, x)


class ResNet50Fashion(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
        self.exit1 = torch.nn.LazyLinear(num_classes)
        self.exit2 = torch.nn.LazyLinear(num_classes)
        self.exit3 = torch.nn.LazyLinear(num_classes)
        self.exit4 = torch.nn.LazyLinear(num_classes)
        self.exit5 = torch.nn.LazyLinear(num_classes)
        self.exit6 = torch.nn.LazyLinear(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit1(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.3404903745651247):
            return (1, y)
        x = self.bn1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit2(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.4218422216176987):
            return (2, y)
        x = self.relu(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit3(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.579648779630661):
            return (3, y)
        x = self.maxpool(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit4(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.561065661907196):
            return (4, y)
        x = self.layer1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit5(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.2823954024910927):
            return (5, y)
        x = self.layer2(x)
        x = self.layer3(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit6(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.2746383741497993):
            return (6, y)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return (0, x)


class ResNet34Cifar10(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
        self.exit1 = torch.nn.LazyLinear(num_classes)
        self.exit2 = torch.nn.LazyLinear(num_classes)
        self.exit3 = torch.nn.LazyLinear(num_classes)
        self.exit4 = torch.nn.LazyLinear(num_classes)
        self.exit5 = torch.nn.LazyLinear(num_classes)
        self.exit6 = torch.nn.LazyLinear(num_classes)
        self.exit7 = torch.nn.LazyLinear(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit1(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.3716316819190979):
            return (1, y)
        x = self.bn1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit2(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.6998863220214844):
            return (2, y)
        x = self.relu(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit3(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.7884806787967682):
            return (3, y)
        x = self.maxpool(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit4(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.8306371507048607):
            return (4, y)
        x = self.layer1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit5(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.9968671581149101):
            return (5, y)
        x = self.layer2(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit6(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.054986223578453):
            return (6, y)
        x = self.layer3(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit7(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.4566168983653187):
            return (7, y)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return (0, x)


class ResNet34Cifar100(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
        self.exit1 = torch.nn.LazyLinear(num_classes)
        self.exit2 = torch.nn.LazyLinear(num_classes)
        self.exit3 = torch.nn.LazyLinear(num_classes)
        self.exit4 = torch.nn.LazyLinear(num_classes)
        self.exit5 = torch.nn.LazyLinear(num_classes)
        self.exit6 = torch.nn.LazyLinear(num_classes)
        self.exit7 = torch.nn.LazyLinear(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit1(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.8547841054201126):
            return (1, y)
        x = self.bn1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit2(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.5203488969802856):
            return (2, y)
        x = self.relu(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit3(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 2.0676015436649324):
            return (3, y)
        x = self.maxpool(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit4(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.761742115020752):
            return (4, y)
        x = self.layer1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit5(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 2.6048172253370288):
            return (5, y)
        x = self.layer2(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit6(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 3.2448585653305053):
            return (6, y)
        x = self.layer3(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit7(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 3.636146520376206):
            return (7, y)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return (0, x)


class ResNet34QMNIST(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
        self.exit1 = torch.nn.LazyLinear(num_classes)
        self.exit2 = torch.nn.LazyLinear(num_classes)
        self.exit3 = torch.nn.LazyLinear(num_classes)
        self.exit4 = torch.nn.LazyLinear(num_classes)
        self.exit5 = torch.nn.LazyLinear(num_classes)
        self.exit6 = torch.nn.LazyLinear(num_classes)
        self.exit7 = torch.nn.LazyLinear(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit1(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.2873958349227905):
            return (1, y)
        x = self.bn1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit2(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.958477783203125):
            return (2, y)
        x = self.relu(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit3(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.5468833446502686):
            return (3, y)
        x = self.maxpool(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit4(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.4222395420074463):
            return (4, y)
        x = self.layer1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit5(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.5985933113098145):
            return (5, y)
        x = self.layer2(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit6(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.4665789444744587):
            return (6, y)
        x = self.layer3(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit7(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.2204003448039293):
            return (7, y)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return (0, x)


class ResNet34Fashion(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
        self.exit1 = torch.nn.LazyLinear(num_classes)
        self.exit2 = torch.nn.LazyLinear(num_classes)
        self.exit3 = torch.nn.LazyLinear(num_classes)
        self.exit4 = torch.nn.LazyLinear(num_classes)
        self.exit5 = torch.nn.LazyLinear(num_classes)
        self.exit6 = torch.nn.LazyLinear(num_classes)
        self.exit7 = torch.nn.LazyLinear(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit1(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.4349217557907106):
            return (1, y)
        x = self.bn1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit2(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.349204912185669):
            return (2, y)
        x = self.relu(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit3(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.3279199600219727):
            return (3, y)
        x = self.maxpool(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit4(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.3156085014343262):
            return (4, y)
        x = self.layer1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit5(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.458837914466858):
            return (5, y)
        x = self.layer2(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit6(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.3115022438764572):
            return (6, y)
        x = self.layer3(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit7(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.5609679216286168):
            return (7, y)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return (0, x)


class ResNet18Cifar10(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
        self.exit1 = torch.nn.LazyLinear(num_classes)
        self.exit2 = torch.nn.LazyLinear(num_classes)
        self.exit3 = torch.nn.LazyLinear(num_classes)
        self.exit4 = torch.nn.LazyLinear(num_classes)
        self.exit5 = torch.nn.LazyLinear(num_classes)
        self.exit6 = torch.nn.LazyLinear(num_classes)
        self.exit7 = torch.nn.LazyLinear(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit1(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.737342455983162):
            return (1, y)
        x = self.bn1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit2(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.7075554728507996):
            return (2, y)
        x = self.relu(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit3(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.759670692384243):
            return (3, y)
        x = self.maxpool(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit4(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.9537064528465271):
            return (4, y)
        x = self.layer1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit5(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.8861846548318862):
            return (5, y)
        x = self.layer2(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit6(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.7517859798669815):
            return (6, y)
        x = self.layer3(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit7(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.5359396690130234):
            return (7, y)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return (0, x)


class ResNet18Cifar100(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
        self.exit1 = torch.nn.LazyLinear(num_classes)
        self.exit2 = torch.nn.LazyLinear(num_classes)
        self.exit3 = torch.nn.LazyLinear(num_classes)
        self.exit4 = torch.nn.LazyLinear(num_classes)
        self.exit5 = torch.nn.LazyLinear(num_classes)
        self.exit6 = torch.nn.LazyLinear(num_classes)
        self.exit7 = torch.nn.LazyLinear(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit1(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.8942746818065643):
            return (1, y)
        x = self.bn1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit2(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 2.3381236064434052):
            return (2, y)
        x = self.relu(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit3(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 2.216015100479126):
            return (3, y)
        x = self.maxpool(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit4(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 2.159404993057251):
            return (4, y)
        x = self.layer1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit5(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 3.0452908778190615):
            return (5, y)
        x = self.layer2(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit6(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 3.435070233345032):
            return (6, y)
        x = self.layer3(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit7(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 4.093396515846252):
            return (7, y)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return (0, x)


class ResNet18QMNIST(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
        self.exit1 = torch.nn.LazyLinear(num_classes)
        self.exit2 = torch.nn.LazyLinear(num_classes)
        self.exit3 = torch.nn.LazyLinear(num_classes)
        self.exit4 = torch.nn.LazyLinear(num_classes)
        self.exit5 = torch.nn.LazyLinear(num_classes)
        self.exit6 = torch.nn.LazyLinear(num_classes)
        self.exit7 = torch.nn.LazyLinear(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit1(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.7255890011787414):
            return (1, y)
        x = self.bn1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit2(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.951309323310852):
            return (2, y)
        x = self.relu(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit3(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.427485911846161):
            return (3, y)
        x = self.maxpool(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit4(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.244315505027771):
            return (4, y)
        x = self.layer1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit5(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.6917553043365479):
            return (5, y)
        x = self.layer2(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit6(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.4168278872966766):
            return (6, y)
        x = self.layer3(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit7(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.2836392819881439):
            return (7, y)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return (0, x)


class ResNet18Fashion(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
        self.exit1 = torch.nn.LazyLinear(num_classes)
        self.exit2 = torch.nn.LazyLinear(num_classes)
        self.exit3 = torch.nn.LazyLinear(num_classes)
        self.exit4 = torch.nn.LazyLinear(num_classes)
        self.exit5 = torch.nn.LazyLinear(num_classes)
        self.exit6 = torch.nn.LazyLinear(num_classes)
        self.exit7 = torch.nn.LazyLinear(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit1(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.5241190099716186):
            return (1, y)
        x = self.bn1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit2(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.0523463213443756):
            return (2, y)
        x = self.relu(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit3(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.394913673400879):
            return (3, y)
        x = self.maxpool(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit4(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.3401701635122298):
            return (4, y)
        x = self.layer1(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit5(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.279452022910118):
            return (5, y)
        x = self.layer2(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit6(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 0.7175035613775254):
            return (6, y)
        x = self.layer3(x)
        y = self.avgpool(x)
        y = y.view(x.size(0), -1)
        y = self.exit7(y)
        pk = F.softmax(y, dim=1)
        entr = -torch.sum(pk * torch.log(pk + 1e-20))
        if torch.all(entr < 1.6668361924588682):
            return (7, y)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return (0, x)


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
