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
from helpers.context_helpers import features as _features
import models.custom_vgg as custom_edit_vgg
import models.custom_resnet as custom_edit_resnet

sys.path.insert(0, os.path.join('external_code', 'EditableNeuralNetworks'))
from lib.editable import Editable, SequentialWithEditable
from lib.utils.ingraph_update import IngraphGradientDescent, IngraphRMSProp


class LeNetModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

        self.features = [self.conv1, self.conv2, self.conv2_drop]

        # Store parameters
        self.model_parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        self.n_params = sum([np.prod(p.size()) for p in self.model_parameters])

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CustomEditLeNet(torch.nn.Sequential):
    def __init__(self,
                 pretrained=False,
                 input_size=(32, 32),
                 in_channels=3,
                 conv_channels=[10, 20],
                 conv_kernel_sizes=[5, 5],
                 max_pool_kernel_sizes=[2, 2],
                 dropouts=[False, True],
                 num_classes=10):

        self.input_size = input_size
        in_channels = [in_channels] + conv_channels[:-1]
        # Sanity checks
        n_convs = len(conv_channels)
        assert len(conv_kernel_sizes) == n_convs
        assert len(max_pool_kernel_sizes) == n_convs
        assert len(dropouts) == n_convs
        assert len(in_channels) == n_convs

        # Create features systematically (probably unnecessary)
        features = []
        output_sizes = [self.input_size]
        for i in range(len(conv_channels)):
            features.append(('conv', torch.nn.Conv2d(
                in_channels[i],
                conv_channels[i],
                kernel_size=conv_kernel_sizes[i]
            )))
            if dropouts[i]:
                features.append(('conv_drop', torch.nn.Dropout2d()))
            features.append(('maxpool', torch.nn.MaxPool2d(
                kernel_size=max_pool_kernel_sizes[i]
            )))

            # Calculate the size of the output
            output_size = [(x - conv_kernel_sizes[i] + 1) // max_pool_kernel_sizes[i] for x in output_sizes[-1]]
            output_sizes.append(output_size)

        # features = [
        #     ('conv', torch.nn.Conv2d(3, conv_channels[0], kernel_size=conv_kernel_sizes[0])),
        #     ('maxpool', torch.nn.MaxPool2d(kernel_size=max_pool_kernel_sizes[0])),
        #     ('conv', torch.nn.Conv2d(10, conv_channels[1], kernel_size=conv_kernel_sizes[1])),
        #     ('conv_drop', torch.nn.Dropout2d()),
        #     ('maxpool', torch.nn.MaxPool2d(kernel_size=max_pool_kernel_sizes[0]))
        # ]

        # Create layer hierarchy to allow for editability (see EditingClassifiers.custom_vgg.py)
        sequence = []
        sequence_dict = {}
        layer_num = -1
        features_layers = []
        for feature_idx, feature in enumerate(features):
            name, function = feature
            if isinstance(function, torch.nn.Conv2d):
                layer_num += 1
                if layer_num > 0:
                    features_layers[-1] = (features_layers[-1][0], torch.nn.Sequential(OrderedDict(features_layers[-1][1])))
                features_layers.append(('layer{}'.format(layer_num), [(name, function)]))
            else:
                features_layers[-1][1].append((name, function))
            sequence_dict['features.{}'.format(feature_idx)] = 'layer{}.{}'.format(layer_num, name)
        features_layers[-1] = (features_layers[-1][0], torch.nn.Sequential(OrderedDict(features_layers[-1][1])))

        # Calculate number of channels for linear layer
        n_channels_latent = output_sizes[-1][0] * output_sizes[-1][1] * conv_channels[-1]

        # Create classifier
        classifier = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(n_channels_latent, 50),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(50, num_classes)
        )

        # Create the sequence usint torch.nn.Sequential
        sequence = features_layers
        sequence.extend([('classifier', classifier)])
        super().__init__(OrderedDict(sequence))

    def forward(self, x):
        for layer in self:  # layers are layer0, layer1, and classifier
            x = layer(x)

        return x

class CIFAR10PretrainedModel(BaseModel):
    '''
    Simple model wrapper for models in external_code/PyTorch_CIFAR10/cifar10_models/state_dicts

    Arg(s):
        type : str
            Name of architecture, must be key in self.all_classifiers

    '''
    def __init__(self,
                 type,
                 checkpoint_path="",
                 device=None):
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
            "inception_v3": inception_v3()
        }
        if type not in self.all_classifiers:
            raise ValueError("Architecture {} not available for pretrained CIFAR-10 models".format(type))
        self.model = self.all_classifiers[type]
        # self.softmax = torch.nn.Softmax(dim=1)

        # Restore weights if checkpoint_path is valid
        self.checkpoint_path = checkpoint_path

        if self.checkpoint_path != "":
            try:
                self.restore_model(checkpoint_path)
            except:
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint)

        # Store parameters
        self.model_parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
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


class ModelWrapperSanturkar(BaseModel):
    def __init__(self,
                 type,
                 layernum=None,
                 checkpoint_path="",
                 device=None,
                 **kwargs):
        super().__init__()

        assert layernum is not None
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
            "lenet": CustomEditLeNet,
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

        self.context_model, self.n_features = _get_context_model(
            model=self.model,
            layernum=self.layernum,
            arch=self.arch)

        self.device = device

        if self.arch.startswith('vgg'):
            self.target_model = self.model[self.layernum + 1]
        elif self.arch.startswith('lenet'):
            self.target_model = self.model[self.layernum]
        else:
            self.target_model = self.model[self.layernum + 1].final

        # Restore checkpoint & convert state dict to be compatible
        self.checkpoint_path = checkpoint_path
        if self.checkpoint_path != "":
            checkpoint = torch.load(checkpoint_path)
            # This should work if model already edited
            try:
                self.model.load_state_dict(checkpoint["state_dict"])
            # If model is not already edited, do key conversion
            except:
                if 'state_dict' in checkpoint.keys():
                    checkpoint = checkpoint['state_dict']
                checkpoint = convert_keys(checkpoint, self.model.state_dict())
                self.model.load_state_dict(checkpoint)

        # Move to cuda
        # self.model = self.model.cuda()
        if device is not None:
            self.model = self.model.to(device)

        # Store parameters
        self.model_parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
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

    def get_feature_values(self):
        return _features

    def get_target_weights(self):
        '''
        Indexing by [0] is because we only want the first element in the list (conv weights)
        Indexing by [1] is because the named_parameters returns tuple of (str, tensor)
        '''
        return list(self.target_model.named_parameters())[0][1].clone()

    def set_device(self, device):
        self.device = device

    def get_device(self):
        return self.device



def convert_keys(checkpoint_state_dict, model_state_dict):
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


class ModelWrapperSinitson(BaseModel):
    def __init__(self,
                 type,
                 optimizer_type,
                 optimizer_args,
                 layernum=None,
                 checkpoint_path="",
                 device=None):

        super().__init__()
        self.all_classifiers = {
            # "vgg11_bn": vgg11_bn(),
            # "vgg13_bn": vgg13_bn(),
            "vgg16_bn": vgg16_bn(),
            # "vgg19_bn": vgg19_bn(),
            "resnet18": resnet18(),
            # "resnet34": resnet34(),
            # "resnet50": resnet50(),
            # "densenet121": densenet121(),
            # "densenet161": densenet161(),
            # "densenet169": densenet169(),
            # "mobilenet_v2": mobilenet_v2(),
            # "googlenet": googlenet(),
            # "inception_v3": inception_v3(),
        }
        self.arch = type
        self.layernum = layernum
        self.features = {}

        # Instantiate model
        assert type in self.all_classifiers.keys()
        self.model = self.all_classifiers[type]

        # Restore weights if checkpoint_path is valid
        self.checkpoint_path = checkpoint_path
        if self.checkpoint_path != "":
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint)

        # Store parameters
        self.model_parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        self.n_params = sum([np.prod(p.size()) for p in self.model_parameters])

        # Move to cuda
        if device is not None:
            self.model = self.model.to(device)
        self.device = device

        # Set optimizer
        if optimizer_type == "RMSProp":
            self.optimizer = IngraphRMSProp(**optimizer_args)
        else:
            self.optimizer = IngraphGradientDescent(**optimizer_args)

    # def get_context_model(self):
    #     def hook_feature(module, input, output):
    #         # self.features['pre'] = input[0]
    #         self.features['post'] = output

    #     if self.layernum is None:
    #         target_layer = model.model.module.features[-1]

    #     target_layer.register_forward_hook(hook_feature)
        # n_features = target_layer[0].in_channels
    def make_editable(self,
                      loss_fn,
                      max_steps=10,
                      ):
        # Function to return model parameters
        get_editable_parameters = lambda module : module.parameters()
        # Function to check if edit is finished (by seeing if loss is <=0)
        is_edit_finished = lambda loss, **kwargs : loss.item() <= 0


        # Make the model editable
        if self.layernum is None:  # edit all layers, pass in the entire module
            self.model = Editable(
                module=self.model,
                loss_function=loss_fn,
                optimizer=self.optimizer,
                max_steps=max_steps,
                get_editable_parameters=get_editable_parameters,
                is_edit_finished=is_edit_finished
            )
        else:
            raise ValueError("SequentialEdit model not yet implemented")

    def edit(self,
             inputs,
             targets,
             max_steps=10,
             model_kwargs=None,
             loss_kwargs=None,
             opt_kwargs=None,
             **kwargs):

        # Move inputs to device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        edit_result = self.model.edit(
            inputs=inputs,
            targets=targets,
            max_steps=max_steps,
            model_kwargs=model_kwargs,
            loss_kwargs=loss_kwargs,
            opt_kwargs=opt_kwargs,
            kwargs=kwargs)

        return edit_result

    def forward(self, x):
        self.logits = self.model(x)
        return self.logits

    def get_checkpoint_path(self):
        return self.checkpoint_path

    def get_n_params(self):
        return self.n_params

    def get_type(self):
        return self.arch

    def set_device(self, device):
        self.device = device

    def get_device(self):
        return self.device