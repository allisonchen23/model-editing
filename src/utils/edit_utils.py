import os, sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, 'src/utils')
from utils import load_image

def prepare_edit_data(key_image_path,
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
        key_image_paths : list[str]
        value_image_paths : list[str]
        mask_paths : None or list[str]
        image_size : None or (int, int) tuple representing (H, W)
            if None, use size of first image

    Returns: edit_data: dict
        edit_data: {
            'imgs': torch.tensor
            'modified_imgs': torch.tensor
            'masks': torch.tensor of masks or torch.ones
        }
    '''
    edit_data = {}
    key_images = []
    value_images = []
    masks = []
    # if mask_path is None:
    #     mask_path = [None for i in range(len(key_image_path))]

    # Load images (and masks if given) and store in lists
    # for key_path, value_path, mask_path in zip(key_image_paths, value_image_paths, mask_paths):
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

    # print(key_images.shape)
    # print(value_images.shape)
    # if len(key_images.shape) == 3:
    #     key_images = torch.stack(key_images, dim=0)
    # if len(value_images.shape) == 3:
    #     value_images = torch.stack(value_images, dim=0)

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
