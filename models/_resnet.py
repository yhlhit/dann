import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    #残差模块，由两个卷基层构成 resnet18 resnet34
    expansion = 1

    def __init__(self,inplanes,planes,stride=1,downsample=None):
        #construct a sample, build layers, 3x3 convolutional layers
        super(BasicBlock,self).__init__()
        self.conv1 = nn.conv2d(inplanes,planes,stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.conv2d(planes,planes,stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride=stride
        #self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
    #use layers for data flow
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    #1x1 3x3 1x1 for resnet 50 or resnet 110
    def __init__(self,inplanes, planes, stride=1,downsample=None):
        super(Bottleneck,self).__init__()

        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,planes*self.expansion,kernel_size =-1,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self,x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            out = self.downsample(out)

        out += residual
        out = self.relu(out)

        return out

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=31):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size= 5)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.conv2 = nn.Conv2d(64, 50, kernel_size= 5)
        # self.bn2 = nn.BatchNorm2d(50)
        self.conv2_drop = nn.Dropout2d()

    def forward(self, input):
        input = input.expand(input.data.shape[0], 3, 28, 28)
        # x = F.relu(F.max_pool2d(self.bn1(self.conv1(input)), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.bn2(self.conv2(x))), 2))
        # x = x.view(-1, 50 * 4 * 4)
        x = F.relu(F.max_pool2d(self.conv1(input), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 48 * 4 * 4)

        return x

class Class_classifier(nn.Module):

    def __init__(self):
        super(Class_classifier, self).__init__()
        # self.fc1 = nn.Linear(50 * 4 * 4, 100)
        # self.bn1 = nn.BatchNorm1d(100)
        # self.fc2 = nn.Linear(100, 100)
        # self.bn2 = nn.BatchNorm1d(100)
        # self.fc3 = nn.Linear(100, 10)
        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, input):
        # logits = F.relu(self.bn1(self.fc1(input)))
        # logits = self.fc2(F.dropout(logits))
        # logits = F.relu(self.bn2(logits))
        # logits = self.fc3(logits)
        logits = F.relu(self.fc1(input))
        logits = self.fc2(F.dropout(logits))
        logits = F.relu(logits)
        logits = self.fc3(logits)

        return F.log_softmax(logits, 1)

class Domain_classifier(nn.Module):

    def __init__(self):
        super(Domain_classifier, self).__init__()
        # self.fc1 = nn.Linear(50 * 4 * 4, 100)
        # self.bn1 = nn.BatchNorm1d(100)
        # self.fc2 = nn.Linear(100, 2)
        self.fc1 = nn.Linear(48 * 4 * 4, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)
        # logits = F.relu(self.bn1(self.fc1(input)))
        # logits = F.log_softmax(self.fc2(logits), 1)
        logits = F.relu(self.fc1(input))
        logits = F.log_softmax(self.fc2(logits), 1)

        return logits



class SVHN_Extractor(nn.Module):

    def __init__(self):
        super(SVHN_Extractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size= 5)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size= 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size= 5, padding= 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3_drop = nn.Dropout2d()
        self.init_params()

    def init_params(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode= 'fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, input):
        input = input.expand(input.data.shape[0], 3, 28, 28)
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 3, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv3_drop(x)

        return x.view(-1, 128 * 3 * 3)

class SVHN_Class_classifier(nn.Module):

    def __init__(self):
        super(SVHN_Class_classifier, self).__init__()
        self.fc1 = nn.Linear(128 * 3 * 3, 3072)
        self.bn1 = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 10)

    def forward(self, input):
        logits = F.relu(self.bn1(self.fc1(input)))
        logits = F.dropout(logits)
        logits = F.relu(self.bn2(self.fc2(logits)))
        logits = self.fc3(logits)

        return F.log_softmax(logits, 1)

class SVHN_Domain_classifier(nn.Module):

    def __init__(self):
        super(SVHN_Domain_classifier, self).__init__()
        self.fc1 = nn.Linear(128 * 3 * 3, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)
        logits = F.relu(self.bn1(self.fc1(input)))
        logits = F.dropout(logits)
        logits = F.relu(self.bn2(self.fc2(logits)))
        logits = F.dropout(logits)
        logits = self.fc3(logits)

        return F.log_softmax(logits, 1)



