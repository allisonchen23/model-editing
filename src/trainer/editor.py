import os, sys

sys.path.insert(0, os.path.join('external_code', 'EditingClassifiers'))
from helpers.context_helpers import get_context_model

class Editor():
    def __init__(self,
                 ntrain,
                 arch,
                 mode_rewrite,
                 layernum,
                 nsteps,
                 lr,
                 restrict_rank=True,
                 nsteps_proj=10,
                 rank=1,
                 use_mask=False):
        '''
        Iniitalize editor with arguments
        '''
        self.layernum = layernum
        self.arch = arch

        self.edit_settings = {
            'ntrain': ntrain,
            'arch': arch,
            'mode_rewrite': mode_rewrite,
            'layernum': layernum,
            'nsteps': nsteps,
            'lr': lr,
            'restrict_rank': restrict_rank,
            'nsteps_proj': nsteps_proj,
            'rank': rank,
            'use_mask': use_mask
        }

    def edit(self,
             train_data,
             context_model,
             val_dataloader=None,
             cache_dir)
    def context_model(self, model):
        return get_context_model(
            model=model,
            layernum=self.layernum,
            arch=self.arch)

    def target_model(self, model):
        if self.arch.startswith('vgg'):
            return model[self.layernum + 1]
        else:
            return model[self.layernum + 1].final