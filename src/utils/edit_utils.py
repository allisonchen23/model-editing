import os, sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, 'src/utils')
from utils import load_image

def prepare_edit_data_eac(key_image_path,
                          value_image_path,
                          mask_path=None,
                          image_size=None):
    '''
    Given
        list of paths to images to form the key ('modified' in the Editing paper)
        list of paths to images to form values ('original' in the Editing paper)
        (opt) list of paths to masks
    Return dictionary

    Arg(s):
        key_image_path : str
        value_image_path : str
        mask_path : None or str
        image_size : None or (int, int) tuple representing (H, W)
            if None, use size of first image

    Returns: edit_data: dict
        edit_data: {
            'imgs': torch.tensor
            'modified_imgs': torch.tensor
            'masks': torch.tensor of masks or torch.ones
        }
    '''

    if image_size is not None:
        assert len(image_size) == 2

    edit_data = {}
    key_images = []
    value_images = []
    masks = []

    # Load images (and masks if given) and store in lists
    key_image = load_image(
        key_image_path,
        data_format='CHW',
        resize=image_size)
    key_image = torch.from_numpy(key_image)

    if image_size is None:
        image_size = (key_image.shape[-2], key_image.shape[-1])

    value_image = load_image(
        value_image_path,
        data_format='CHW',
        resize=image_size)

    value_image = torch.from_numpy(value_image)

    key_images.append(key_image)
    value_images.append(value_image)

    if mask_path is not None:
        mask = torch.from_numpy(np.load(mask_path))
        if mask.shape[-2:] != image_size:
            mask = torch.nn.functional.interpolate(
                mask,
                size=image_size)
    else:
        mask = torch.ones_like(key_image)
    masks.append(mask)

    # Convert lists to tensors
    key_images = torch.stack(key_images, dim=0)
    value_images = torch.stack(value_images, dim=0)
    if masks[0] is not None:
        masks = torch.stack(masks, dim=0)

    # Store in dictionary
    edit_data['imgs'] = value_images
    edit_data['modified_imgs'] = key_images
    edit_data['masks'] = masks

    return edit_data

def get_target_weights(target_model):
    '''
    Copied from EditingClassifiers/helpers/rewrite_helpers.py
    '''
    return [p for n, p in target_model.named_parameters()
            if 'weight' in n][0]


def prepare_edit_data_enn(edit_image_paths,
                          edit_labels,
                          image_size):
    edit_images = []
    for edit_image_path in edit_image_paths:
        edit_images.append(load_image(
            edit_image_path,
            data_format='CHW',
            image_size=image_size
        ))
    return None

def prepare_edit_data(edit_method : str, **kwargs):
    '''
    Given an editing method and necessary key word arguments, prepare the edit data

    Arg(s):
        edit_method : str
            the editing method used
        kwargs : dict
            necessary keyword arguments
    '''
    if edit_method == 'eac':
        return prepare_edit_data_eac(**kwargs)
    elif edit_method == 'enn':
        return prepare_edit_data_enn(**kwargs)
    else:
        raise ValueError("Edit method '{}' not supported.".format(edit_method))
