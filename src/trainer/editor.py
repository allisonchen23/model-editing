import os, sys
from argparse import Namespace
sys.path.insert(0, os.path.join('external_code', 'EditingClassifiers'))
from helpers.context_helpers import get_context_model
from helpers.rewrite_helpers import edit_classifier

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

        # return context_model

    # def _context_model(self, model):
    #     return get_context_model(
    #         model=model,
    #         layernum=self.layernum,
    #         arch=self.arch)

    # def _target_model(self, model):
    #     if self.arch.startswith('vgg'):
    #         return model[self.layernum + 1]
    #     else:
    #         return model[self.layernum + 1].final

    # def get_layernum(self):
    #     return self.layernum
