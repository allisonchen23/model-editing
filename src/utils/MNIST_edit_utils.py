import os, sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, 'src/utils')
from utils import load_image
sys.path.insert(0, 'src')
import datasets.datasets as module_data

def prepare_edit_data_eac(keys, values, labels):
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
#     edit_idxs_path = os.path.join(data_dir, edit_idxs_file)
#     edit_pool_path = os.path.join(data_dir, edit_pool_file)
    
#     edit_idxs = torch.load(
#     if image_size is not None:
#         assert len(image_size) == 2

    edit_data = {}
#     key_images = []
#     value_images = []
#     masks = []

#     # Load images (and masks if given) and store in lists
#     key_image = load_image(
#         key_image_path,
#         data_format='CHW',
#         resize=image_size)
#     key_image = torch.from_numpy(key_image)

#     if image_size is None:
#         image_size = (key_image.shape[-2], key_image.shape[-1])

#     value_image = load_image(
#         value_image_path,
#         data_format='CHW',
#         resize=image_size)

#     value_image = torch.from_numpy(value_image)

#     key_images.append(key_image)
#     value_images.append(value_image)

#     if mask_path is not None:
#         mask = torch.from_numpy(np.load(mask_path))
#         if mask.shape[-2:] != image_size:
#             mask = torch.nn.functional.interpolate(
#                 mask,
#                 size=image_size)
#     else:
#         mask = torch.ones_like(key_image)
#     masks.append(mask)

#     # Convert lists to tensors
#     key_images = torch.stack(key_images, dim=0)
#     value_images = torch.stack(value_images, dim=0)
#     if masks[0] is not None:
#         masks = torch.stack(masks, dim=0)
    pad_fn = transforms.Pad(padding=2)
    key_images = pad_fn(torch.from_numpy(keys))
    value_images = pad_fn(torch.from_numpy(values))
    print(key_images.shape, value_images.shape)
    non_black = torch.sum(key_images, axis=1, keepdim=True)
    # masks = torch.where(non_black > 0, 1, 0).to(torch.int32)
    
    mask_shape = list(key_images.shape)
    mask_shape[1] = 1
    masks = torch.ones(mask_shape)
    print(masks.shape)

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
                          image_size,
                          # dataset args
                          normalize=False,
                          means=None,
                          stds=None,
                          #data loader args
                          batch_size=256,
                          shuffle=False,
                          num_workers=8):
    '''
    Given parameters for a data loader, return a data loader of the edit images
    '''

    # edit_data_loader = torch.utils.data.DataLoader(
    #     module_data.CINIC10Dataset(
    #         data_dir="",
    #         image_paths=edit_image_paths,
    #         labels=edit_labels,
    #         return_paths=False,
    #         normalize=normalize,
    #         means=means,
    #         stds=stds),
    #     batch_size=batch_size,
    #     shuffle=shuffle,
    #     num_workers=num_workers)


    # return edit_data_loader
    edit_images = []
    for edit_image_path in edit_image_paths:
        edit_images.append(load_image(edit_image_path, data_format='CHW', resize=image_size))

    edit_images = np.stack(edit_images, axis=0)
    edit_images = torch.from_numpy(edit_images)
    edit_labels = torch.tensor(edit_labels)
    return edit_images, edit_labels

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
