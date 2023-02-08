import os, sys
import torch
from argparse import Namespace
sys.path.insert(0, os.path.join('external_code', 'EditingClassifiers'))
from helpers.context_helpers import get_context_model
from helpers.rewrite_helpers import edit_classifier

sys.path.insert(0, 'src')
from utils.edit_utils import get_target_weights

class Editor():
    def __init__(self,
                 ntrain,
                 arch,
                 mode_rewrite,
                 nsteps,
                 lr,
                 restrict_rank=True,
                 nsteps_proj=10,
                 rank=1,
                 use_mask=False):
        '''
        Iniitalize editor with arguments
        '''
        self.arch = arch

        self.edit_settings = {
            'ntrain': ntrain,
            'arch': arch,
            'mode_rewrite': mode_rewrite,
            'nsteps': nsteps,
            'lr': lr,
            'restrict_rank': restrict_rank,
            'nsteps_proj': nsteps_proj,
            'rank': rank,
            'use_mask': use_mask
        }
        self.edit_settings = Namespace(**self.edit_settings)

        self.edit_data = None
        self.weight_diff = None

    def edit(self,
             edit_data,
             model,
             val_data_loader,
             cache_dir=None):

        self.val_data_loader = val_data_loader

        self.edit_data = edit_data
        layernum = model.layernum


        context_model = model.context_model
        target_model = model.target_model

        original_weights = get_target_weights(target_model).clone()

        if self.val_data_loader is not None:
            key_method = 'zca'
        else:
            # Obtain number of input features for covariance matrix
            if self.arch.startswith('vgg'):
                n_features = model.model[layernum + 1][0].in_channels
            elif self.arch.startswith('clip'):
                n_features = model.model.visual[layernum + 1].final.conv3.module.in_channels
            elif self.arch == 'resnet50':
                n_features = model.model[layernum + 1].final.conv3.module.in_channels
            elif self.arch == 'resnet18':
                n_features = model.model[layernum + 1].final.conv2.module.in_channels

            key_method = n_features

        self.context_model = edit_classifier(
            args=self.edit_settings,
            # layernum=layernum,
            train_data=edit_data,
            context_model=context_model,
            target_model=target_model,
            val_loader=self.val_data_loader,
            key_method=key_method,
            caching_dir=cache_dir)

        edited_weights = get_target_weights(target_model).clone()
        self.weight_diff = edited_weights - original_weights


    def random_edit(self,
                    model,
                    noise_mean=0.0,
                    noise_std=0.003):
        '''
        Given a model and noise parameters, edit the target layer by with noise

        Arg(s):

        Returns:
        '''

        device = model.get_device()

        target_model = model.target_model
        target_weights = get_target_weights(target_model)
        original_weights = target_weights.clone()

        shape = target_weights.shape

        edit_weights = (torch.randn(size=shape) * noise_std) + noise_mean
        if device is not None:
            edit_weights = edit_weights.to(device)
        with torch.no_grad():
            target_weights[...] = target_weights + edit_weights

        print("L2 norm of weight change: {}".format(torch.norm(target_weights - original_weights).item()))
        post_edit_weights = get_target_weights(target_model)
        print("pre_edit_weights id: {} mean: {}".format(id(original_weights), torch.mean(original_weights)))
        print("post_edit_weights id: {} mean: {}".format(id(post_edit_weights), torch.mean(post_edit_weights)))
        print("are the tensors equal: {}".format((post_edit_weights == original_weights).all()))

    def get_weight_diff(self):
        if self.weight_diff is None:
            raise ValueError("Unable to obtain weight difference without editing model")
        else:
            return self.weight_diff

    # def bump_edit(self,
    #               model,
    #               target_class_idx,
    #               bump_amount,
    #               random_parameters=(None, None)):

    #     def forward(self, x):
    #         self.logits = self.model(x)
    #         self.logits[target_class_idx] += bump_amount
    #         return self.logits

    #     model.forward = forward

