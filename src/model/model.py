import sys, os
sys.path.insert(0, os.path.join('external_code', 'PyTorch_CIFAR10'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CIFAR10PretrainedModel(BaseModel):
    def __init__(self, type, checkpoint_path=""):
        super().__init__()
        self.all_classifiers = {
            "vgg11_bn": vgg11_bn(),
            "vgg13_bn": vgg13_bn(),
            "vgg16_bn": vgg16_bn(),
            "vgg19_bn": vgg19_bn(),
            "resnet18": resnet18(),
            "resnet34": resnet34(),
            "resnet50": resnet50(),
            "densenet121": densenet121(),
            "densenet161": densenet161(),
            "densenet169": densenet169(),
            "mobilenet_v2": mobilenet_v2(),
            "googlenet": googlenet(),
            "inception_v3": inception_v3(),
        }
        if type not in self.all_classifiers:
            raise ValueError("Architecture {} not available for pretrained CIFAR-10 models".format(type))
        self.model = self.all_classifiers[type]
        self.softmax = torch.nn.Softmax(dim=1)
        if checkpoint_path != "":
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint)

    def forward(self, x):
        return self.softmax(self.model(x))
