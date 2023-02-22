import os, sys
import torch
from argparse import Namespace
sys.path.insert(0, os.path.join('external_code', 'EditingClassifiers'))
from helpers.context_helpers import get_context_model
from helpers.rewrite_helpers import edit_classifier

sys.path.insert(0, os.path.join('external_code', 'EditableNeuralNetworks'))
from lib.trainer import EditableTrainer, DistillationEditableTrainer
from lib.evaluate import calculate_edit_statistics, evaluate_quality
from lib.utils import training_mode

sys.path.insert(0, 'src')
from utils.edit_utils import get_target_weights

class EditorEAC():
    def __init__(self,
                 ntrain,
                 arch,
                 mode_rewrite,
                 nsteps,
                 lr,
                 restrict_rank=True,
                 nsteps_proj=10,
                 rank=1,
                 use_mask=False,
                 noise_edit=False):
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

        self.noise_edit = noise_edit

    # def setup_editor(covariance_image_paths=None,
    #                  covariance_labels=None):
    #     if covariance_image_paths is not None and covariance labels is not None:

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


    def noise_edit(self,
                    model,
                    noise_mean=0.0,
                    noise_std=0.003):
        '''
        Given a model and noise parameters, edit the target layer by with noise

        Arg(s):
            model : torch.nn.Module
                model to edit. Must have property target_model
                ModelWrapperSanturkar
            noise_mean : float
                mean of Gaussian noise parameter
            noise_std : float
                STD of Gaussian noise parameter
        Returns:
            None
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


class EditorENN():
    def __init__(self,
                 noise_edit):
        pass

    def edit(self,
             model,
             inputs,
             targets):

        return model.edit(
            inputs=inputs,
            targets=targets)

    # def __init__(self,
    #              model,
    #              loss_fn,
    #              optimizer,
    #              stability_coefficient,
    #              editability_coefficient,
    #              max_norm,
    #              error_fn=None,
    #              noise_edit=False):
    #     self.editable_model = model.model
    #     self.model = model
    #     # model_parameters = model.
    #     if loss_fn is None:
    #         if error_fn is None:
    #             self.trainer = DistillationEditableTrainer(
    #                 model=self.editable_model,
    #                 optimizer=optimizer,
    #                 stability_coeff=stability_coefficient,
    #                 editability_coeff=editability_coefficient,
    #                 max_norm=max_norm
    #             )
    #         else:
    #             self.trainer = DistillationEditableTrainer(
    #                 model=self.editable_model,
    #                 error_function=error_fn,
    #                 optimizer=optimizer,
    #                 stability_coeff=stability_coefficient,
    #                 editability_coeff=editability_coefficient,
    #                 max_norm=max_norm
    #             )
    #     else:
    #         self.trainer = EditableTrainer(
    #                 model=self.editable_model,
    #                 loss_function=loss_fn,
    #                 optimizer=optimizer,
    #                 stability_coeff=stability_coefficient,
    #                 editability_coeff=editability_coefficient,
    #                 max_norm=max_norm
    #             )
    # def edit(self,
    #          val_data_loader,
    #          edit_data_loader,
    #          **kwargs):
    #     edit_images, edit_labels = map(torch.cat, zip(*edit_data_loader))
    #     print(type(edit_images))

    #     device = self.model.device
    #     edit_images = edit_images.to(device)
    #     edit_labels = edit_labels.to(device)
    #     # Copied from claculate_edit_statistics() in lib/
    #     # progressbar = tqdm if progressbar is True else progressbar or nop
    #     results_temporary = []

    #     with training_mode(self.model.model, is_train=False):
    #         for i in range(len(edit_images)):
    #             edited_model, success, loss, complexity = self.editable_model.edit(
    #                 edit_images[i:i + 1], edit_labels[i:i + 1], detach=True, **kwargs)
    #             # results_temporary.append((error_function(edited_model, X_test, y_test), success, complexity))
    #     return edited_model # results_temporary

    #     # device = self.model.device

    #     # # val_images, val_labels, _ = map(torch.cat, zip(*val_data_loader))
    #     # val_images, val_labels, _ = next(iter(val_data_loader))
    #     # print(type(val_images))
    #     # edit_images, edit_labels = map(torch.cat, zip(*edit_data_loader))
    #     # self.trainer.step(
    #     #     x_batch=val_images.to(device),
    #     #     y_batch=val_labels.to(device),
    #     #     x_edit=edit_images.to(device),
    #     #     y_edit=edit_labels.to(device)
    #     # )


def get_editor(edit_method, **kwargs):
    '''
    Given an editing method and necessary key word arguments, return the appropriate Editor

    Arg(s):
        edit_method : str
            the editing method used
        kwargs : dict
            necessary keyword arguments
    '''
    if edit_method == 'eac':
        return EditorEAC(**kwargs)
    elif edit_method == 'enn':
        return EditorENN(**kwargs)
    else:
        raise ValueError("Edit method '{}' not supported.".format(edit_method))