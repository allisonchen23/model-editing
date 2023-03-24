import torch
import numpy as np
import sys

from torchvision import transforms
from torch.utils.data import Dataset

sys.path.insert(0, 'src')
from utils import read_lists

class MNISTEACDataset(Dataset):
    '''
    Given the pool of edit data and the indices for images for each edit
        give a batch of data

    Arg(s):
        edit_pool_path : str
            path to data stored as dict
                {
                    'keys': N x 3 x H x W np.array
                    'values': N x 3 x H x W np.array
                    'labels': N np.array,
                    'test_set_idxs': N np.array (which indices of test set these values come from),
                    'masks': N x 1 x H x W np.array or None
                }
    '''

    def __init__(self,
                 edit_pool_path: str,
                 edit_idxs_path: str,
                 use_masks:bool=True,
                 padding: int=0,
                 normalize: bool=False,
                 means: list=None,
                 stds: list=None,
                 target_transform=None):
        # Load in edit pool
        self.edit_pool = torch.load(edit_pool_path)
        # Load in and process list of indices to use for each edit
        self.edit_idxs = read_lists(edit_idxs_path)
        self.edit_idxs_path = edit_idxs_path

        # convert each string to list
        for row_idx, row in enumerate(self.edit_idxs):
            row = row.split(',')
            row = [eval(i) for i in row]
            self.edit_idxs[row_idx] = np.array(row)

        # Assert masks exist if using them
        self.use_masks = use_masks
        if self.use_masks:
            assert 'masks' in self.edit_pool.keys()

        # Separate keys, values, masks and convert all of them to torch.Tensors
        self.keys = torch.from_numpy(self.edit_pool['keys'])
        self.values = torch.from_numpy(self.edit_pool['values'])
        self.masks = torch.from_numpy(self.edit_pool['masks'])
        if 'labels' in self.edit_pool.keys():
            self.labels = torch.from_numpy(self.edit_pool['labels'])
        else:
            self.labels = None

        # Create transforms
        geometric_transform = []
        pixel_transform = []
        if padding > 0:
            geometric_transform.append(transforms.Pad(padding, padding_mode='edge'))
        if normalize:
            assert means is not None and stds is not None, "Cannot normalize without means and stds"
            pixel_transform.append(transforms.Normalize(mean=means, std=stds))
        self.geometric_transform = transforms.Compose(geometric_transform)
        self.pixel_transform = transforms.Compose(pixel_transform)


    def __getitem__(self, index):
        cur_idxs = self.edit_idxs[index] # list of indices
        cur_keys = self.keys[cur_idxs]
        cur_values = self.values[cur_idxs]
        if self.labels is not None:
            cur_labels = self.labels[cur_idxs]
        else:
            cur_labels = -1

        if self.use_masks:
            cur_masks = self.masks[cur_idxs]
        else:
            masks_shape = list(cur_keys.shape)
            masks_shape[-3] = 1
            cur_masks =  torch.ones(masks_shape)

        cur_masks = cur_masks.to(torch.int32)

        cur_keys = self.pixel_transform(self.geometric_transform(cur_keys))
        cur_values = self.pixel_transform(self.geometric_transform(cur_values))
        cur_masks = self.geometric_transform(cur_masks)



        # This is the format EAC expects
        cur_edit_data = {
            'imgs': cur_values,
            'modified_imgs': cur_keys,
            'masks': cur_masks
        }

        return cur_edit_data, cur_labels

    def __len__(self):
        return len(self.edit_idxs)

class MNISTENNDataset(Dataset):
    '''
    Given the pool of edit data and the indices for images for each edit
        give a batch of data

    Arg(s):
        edit_pool_path : str
            path to data stored as dict
                {
                    'keys': N x 3 x H x W np.array
                    'values': N x 3 x H x W np.array
                    'labels': N np.array,
                    'test_set_idxs': N np.array (which indices of test set these values come from),
                    'masks': N x 1 x H x W np.array or None
                }
    '''

    def __init__(self,
                 edit_pool_path: str,
                 edit_idxs_path: str,
                #  use_masks:bool=True,
                 padding: int=0,
                 normalize: bool=False,
                 means: list=None,
                 stds: list=None,
                 target_transform=None):
        # Load in edit pool
        self.edit_pool = torch.load(edit_pool_path)
        # Load in and process list of indices to use for each edit
        self.edit_idxs = read_lists(edit_idxs_path)
        self.edit_idxs_path = edit_idxs_path

        # convert each string to list
        for row_idx, row in enumerate(self.edit_idxs):
            row = row.split(',')
            row = [eval(i) for i in row]
            self.edit_idxs[row_idx] = np.array(row)

        # Assert masks exist if using them
        # self.use_masks = use_masks
        # if self.use_masks:
        #     assert 'masks' in self.edit_pool.keys()

        # Separate keys, values, masks and convert all of them to torch.Tensors
        self.images = torch.from_numpy(self.edit_pool['keys'])
        # self.values = torch.from_numpy(self.edit_pool['values'])
        # self.masks = torch.from_numpy(self.edit_pool['masks'])
        if 'labels' not in self.edit_pool.keys():
            raise ValueError("Missing key 'labels' in edit pool. Required for ENN")
        else:
            self.labels = torch.from_numpy(self.edit_pool['values'])

        # Create transforms
        geometric_transform = []
        pixel_transform = []
        if padding > 0:
            geometric_transform.append(transforms.Pad(padding, padding_mode='edge'))
        if normalize:
            assert means is not None and stds is not None, "Cannot normalize without means and stds"
            pixel_transform.append(transforms.Normalize(mean=means, std=stds))
        self.geometric_transform = transforms.Compose(geometric_transform)
        self.pixel_transform = transforms.Compose(pixel_transform)


    def __getitem__(self, index):
        cur_idxs = self.edit_idxs[index] # list of indices
        cur_images = self.keys[cur_idxs]
        # cur_values = self.values[cur_idxs]
        cur_labels = self.labels[cur_idxs]
        # if self.labels is not None:
        #     cur_labels = self.labels[cur_idxs]
        # else:
        #     cur_labels = -1

        # if self.use_masks:
        #     cur_masks = self.masks[cur_idxs]
        # else:
        #     masks_shape = list(cur_keys.shape)
        #     masks_shape[-3] = 1
        #     cur_masks =  torch.ones(masks_shape)

        # cur_masks = cur_masks.to(torch.int32)

        cur_images = self.pixel_transform(self.geometric_transform(cur_images))
        # cur_values = self.pixel_transform(self.geometric_transform(cur_values))
        # cur_masks = self.geometric_transform(cur_masks)



        # This is the format EAC expects
        # cur_edit_data = {
        #     'imgs': cur_values,
        #     'modified_imgs': cur_keys,
        #     'masks': cur_masks
        # }

        return cur_images, cur_labels

    def __len__(self):
        return len(self.edit_idxs)