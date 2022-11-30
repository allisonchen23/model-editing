import sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from base import BaseModel
sys.path.insert(0, os.path.join('external_code', 'PyTorch_CIFAR10'))
from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
sys.path.insert(0, os.path.join('external_code', 'EditingClassifiers'))
from helpers.context_helpers import get_context_model as _get_context_model
import models.custom_vgg as custom_edit_vgg
import models.custom_resnet as custom_edit_resnet


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

        # Restore weights if checkpoint_path is valid
        self.checkpoint_path = checkpoint_path
        if self.checkpoint_path != "":
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint)

        # Store parameters
        self.model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.n_params = sum([np.prod(p.size()) for p in self.model_parameters])

    def forward(self, x):
        self.logits = self.model(x)
        return self.logits

    def get_features(self, x):
        features = self.model.features(x)
        return features

    def get_checkpoint_path(self):
        return self.checkpoint_path

    def get_n_params(self):
        return self.n_params


class CIFAR10PretrainedModelEdit(BaseModel):
    def __init__(self, type, layernum, checkpoint_path="",**kwargs):
        super().__init__()
        self.all_classifiers = {
            # "vgg11_bn": vgg11_bn(),
            # "vgg13_bn": vgg13_bn(),
            "vgg16_bn": custom_edit_vgg.vgg16_bn,
            "vgg16": custom_edit_vgg.vgg16,
            # "vgg19_bn": vgg19_bn(),
            "resnet18": custom_edit_resnet.resnet18,
            # "resnet34": resnet34(),
            "resnet50": custom_edit_resnet.resnet50,
            # "densenet121": densenet121(),
            # "densenet161": densenet161(),
            # "densenet169": densenet169(),
            # "mobilenet_v2": mobilenet_v2(),
            # "googlenet": googlenet(),
            # "inception_v3": inception_v3(),
        }
        self.arch = type
        assert type in self.all_classifiers.keys()
        if 'mean' in kwargs:
            kwargs['mean'] = torch.tensor(kwargs['mean'])
        if 'std' in kwargs:
            kwargs['std'] = torch.tensor(kwargs['std'])
        self.layernum = layernum
        # Build model, obtain context_model (with hooks) and target_model (just layer to edit)
        self.model = self.all_classifiers[type](pretrained=False, **kwargs)
        self.context_model = _get_context_model(
            model=self.model,
            layernum=self.layernum,
            arch=self.arch)
        if self.arch.startswith('vgg'):
            self.target_model = self.model[self.layernum + 1]
        else:
            self.target_model = self.model[self.layernum + 1].final

        # Restore checkpoint & convert state dict to be compatible
        self.checkpoint_path = checkpoint_path
        if self.checkpoint_path != "":
            checkpoint = torch.load(checkpoint_path)
            checkpoint = convert_keys_vgg(checkpoint, self.model.state_dict())
            self.model.load_state_dict(checkpoint)

        # Move to cuda
        self.model = self.model.cuda()
        # Switch to evaluation mode
        # self.model.eval()

        # Store parameters
        self.model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.n_params = sum([np.prod(p.size()) for p in self.model_parameters])

    def forward(self, x):
        self.logits = self.model(x)
        return self.logits

    def get_checkpoint_path(self):
        return self.checkpoint_path

    def get_n_params(self):
        return self.n_params

    def get_type(self):
        return self.arch

    # def get_context_model(self):
    #     return self.context_model

    # def


def convert_keys_vgg(checkpoint_state_dict, model_state_dict):
    '''
    Given a checkpoint_state_dict, convert the keys to match the model_state_dict
    Arg(s):
        checkpoint_state_dict : OrderedDict
            the state dictionary of the checkpoint
        model_state_dict : OrderedDict
            the state dictionary of the model from Editing a Classifier
    Returns:
        new_state_dict : OrderedDict
            keys from model_state_dict but values from checkpoint_state_dict
    '''
    assert len(checkpoint_state_dict.keys()) == len(model_state_dict.keys()), \
        "Unequal state dict lengths. Checkpoint length: {} Model length: {}".format(
            len(checkpoint_state_dict.keys()), len(model_state_dict.keys()))
    new_state_dict = OrderedDict()
    for (model_key, model_val), (ckpt_key, ckpt_val) in zip(model_state_dict.items(), checkpoint_state_dict.items()):
        # Check that corresponding keys have the same tensor shape
        assert model_val.shape == ckpt_val.shape, \
            "Not same shapes. Model[{}]: {} Checkpoint[{}]: {}".format(
                model_key, model_val.shape, ckpt_key, ckpt_val.shape)
        new_state_dict[model_key] = ckpt_val

    return new_state_dict


