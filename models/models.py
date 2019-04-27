import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as tmodels
import torch.nn.init as init

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

class Small_Domain_classifier(nn.Module):

    def __init__(self):
        super(Small_Domain_classifier, self).__init__()
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

class Res50_Extractor(nn.Module):
    def __init__(self, use_bottleneck=True,bottleneck_dim=256):
        super(Res50_Extractor, self).__init__()
        model_resnet50 = tmodels.resnet50(pretrained=True)

        #first stage
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool

        #layer1~4
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4

        self.avgpool = model_resnet50.avgpool

        self.feature = nn.Sequential(self.conv1,self.bn1,self.relu,self.maxpool,self.layer1,self.layer2,self.layer3,self.layer4,self.avgpool)

        self.use_bottleneck = use_bottleneck
        self.bottleneck_dim = bottleneck_dim

        if self.use_bottleneck:
            #bottleneck_module = nn.Linear(model_resnet50.fc.in_features,bottleneck_dim)
            bottleneck_module = nn.Linear(model_resnet50.fc.in_features, bottleneck_dim)
            bottleneck_module.weight.data.normal_(0,0.0005)
            bottleneck_module.bias.data.fill_(0.0)

            self.bottleneck = bottleneck_module

    def forward(self, input):
        x = self.feature(input)
        x = x.view(x.size(0),-1)
        if self.use_bottleneck:
            x = self.bottleneck(x)

        return x

class Res50_Class_classifier(nn.Module):

    def __init__(self,use_bottleneck=True,bottleneck_dim=256,class_num=10):
        super(Res50_Class_classifier, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.bottleneck_dim = bottleneck_dim

        if self.use_bottleneck:
            inputs_dim = self.bottleneck_dim
        else:
            model_resnet50 = tmodels.resnet50(pretrained=False)
            inputs_dim = model_resnet50.fc.in_features
        self.classifier = nn.Linear(inputs_dim,class_num)
        #self.classifier.weight.data.normal_(0,0.01)
        #self.classifier.bias.data.fill_(0.0)


    def forward(self, input):

        #logits = F.relu(self.fc1(input))
        #logits = self.fc2(F.dropout(logits))
        #logits = F.relu(logits)
        #logits = self.fc3(logits)
        logits = self.classifier(input)
        logits = F.log_softmax(logits,1)

        return logits

class Domain_classifier(nn.Module):

    def __init__(self,use_bottleneck=True,bottleneck_dim=256):
        super(Domain_classifier, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.bottleneck_dim = bottleneck_dim

        if self.use_bottleneck:
            inputs_dim = self.bottleneck_dim
        else:
            model_resnet50 = tmodels.resnet50(pretrained=False)
            inputs_dim = model_resnet50.fc.in_features

        self.dc_fc1 = nn.Linear(inputs_dim, 1024)
        self.dc_fc1.weight.data.normal_(0.0,0.01)
        self.dc_fc1.bias.data.fill_(0.0)
        self.dc_fc2 = nn.Linear(1024, 1024)
        self.dc_fc2.weight.data.normal_(0.0, 0.01)
        self.dc_fc2.bias.data.fill_(0.0)
        self.dc_fc3 = nn.Linear(1024,2)
        self.dc_fc3.weight.data.normal_(0.0, 0.3)
        self.dc_fc3.bias.data.fill_(0.0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.domain_classifier = nn.Sequential(self.dc_fc1,self.relu,self.dropout,self.dc_fc2,self.relu,self.dropout,self.dc_fc3)


    def forward(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)
        logits = self.domain_classifier(input)
        logits = F.log_softmax(logits, 1)

        return logits
