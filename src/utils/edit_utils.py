import os,sys
import torch
import numpy as np
from PIL import Image


def prepare_edit_data(key_image_paths,
                      value_image_paths,
                      mask_paths=None):
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

    Returns: edit_data: dict
        edit_data: {
            'imgs': torch.tensor
            'modified_imgs': torch.tensor
            'masks': torch.tensor or None
        }
    '''
    edit_data = {}
    key_images = []
    value_images = []
    if mask_paths is not None:
        masks = []
    else:
        masks = None
        mask_paths = [None for i in range(len(key_image_paths))]

    # Load images (and masks if given) and store in lists
    for key_path, value_path, mask_path in zip(key_image_paths, value_image_paths, mask_paths):
        key_image = read_image(key_path, as_tensor=True)
        value_image = read_image(value_path, as_tensor=True)

        key_images.append(key_image)
        value_images.append(value_image)
        if mask_path is not None:
            mask = torch.from_numpy(np.load(mask_path))
            masks.append(mask)

    # Convert lists to tensors
    key_images = torch.stack(key_images, axis=0)
    value_images = torch.stack(value_images, axis=0)
    if masks is not None:
        masks = torch.stack(masks, axis=0)

    # Store in dictionary
    edit_data['imgs'] = value_images
    edit_data['modified_imgs'] = key_images
    edit_data['masks'] = masks

    return edit_data


def read_image(path, as_tensor=False):
    image = np.array(Image.open(path)).astype('float32')
    image = np.transpose(image, (2, 0, 1))
    if not as_tensor:
        return image
    else:
        return torch.from_numpy(image)
