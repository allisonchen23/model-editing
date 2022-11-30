import os, sys
from argparse import Namespace
sys.path.insert(0, os.path.join('external_code', 'EditingClassifiers'))
from helpers.context_helpers import get_context_model
from helpers.rewrite_helpers import edit_classifier

class Editor():
    def __init__(self,
                 model,
                 val_data_loader,
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
        # self.layernum = layernum
        self.arch = arch

        # Get model info
        self.model = model  #CIFAR10PretrainedModelEdit wrapper
        self.context_model, _ = self._context_model(model.model)
        self.target_model = self._target_model(model.model)

        self.val_data_loader = val_data_loader

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
             cache_dir=None):
        self.edit_data = edit_data

        layernum = model.layernum
        context_model = model.context_model
        target_model = model.target_model

        self.context_model = edit_classifier(
            args=self.edit_settings,
            layernum=layernum,
            train_data=edit_data,
            context_model=context_model,
            target_model=target_model,
            val_loader=self.val_data_loader,
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
