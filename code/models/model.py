from torchvision import models
import torch.nn as nn
import torch as t
from models.Basicmodule import Basicmodule
# import segmentation_models_pytorch as smp


class DenseNet121(Basicmodule):
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.classifier.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output


class DenseNet169(Basicmodule):
    def __init__(self):
        super(DenseNet169, self).__init__()
        self.model = models.densenet169(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(1664, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.classifier.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class DenseNet201(Basicmodule):
    def __init__(self):
        super(DenseNet201, self).__init__()
        self.model = models.densenet201(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(1920, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.classifier.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class MobileNet(Basicmodule):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.classifier.parameters()):
            if index >= 1:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class VGG16(Basicmodule):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier[6] = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.classifier.parameters()):
            if index >= 6:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class VGG19(Basicmodule):
    def __init__(self):
        super(VGG19, self).__init__()
        self.model = models.vgg19(pretrained=True)
        self.model.classifier[6] = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.classifier.parameters()):
            if index >= 6:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class ResNet34(Basicmodule):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.fc.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class ResNet50(Basicmodule):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.fc.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class ResNet101(Basicmodule):
    def __init__(self):
        super(ResNet101, self).__init__()
        self.model = models.resnet101(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.fc.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class ResNet152(Basicmodule):
    def __init__(self):
        super(ResNet152, self).__init__()
        self.model = models.resnet152(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.fc.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class ResNeXt50(Basicmodule):
    def __init__(self):
        super(ResNeXt50, self).__init__()
        self.model = models.resnext50_32x4d(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.fc.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class ResNeXt101(Basicmodule):
    def __init__(self):
        super(ResNeXt101, self).__init__()
        self.model = models.resnext101_32x8d(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.fc.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output

class InceptionV3(Basicmodule):
    def __init__(self):
        super(InceptionV3, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        self.model.AuxLogits.fc = nn.Linear(768, 2)
        self.model.aux_logits = False
        for params in self.model.parameters():
            params.requires_grad = False
        for index, param in enumerate(self.model.fc.parameters()):
            if index >= 0:
                param.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output
#
# net = DenseNet121()
# print(net)