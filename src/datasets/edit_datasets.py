import torch
import numpy as np
import sys

from torchvision import transforms
from torch.utils.data import Dataset

sys.path.insert(0, 'src')
from utils import read_lists

def MNISTEACDataset(Dataset):
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
                 use_masks:bool =True,
                 padding: int=0):
        # Load in edit pool
        self.edit_pool = torch.load(edit_pool_path)
        # Load in and process list of indices to use for each edit
        self.edit_idxs = read_lists(edit_idxs_path)

        # convert each string to list
        for row_idx, row in enumerate(self.edit_idxs):
            row = row.split(',')
            row = [eval(i) for i in row]
            self.edit_idxs[row_idx] = np.array(row)

        # Assert masks exist if using them
        self.use_masks = use_masks
        if self.use_masks:
            assert 'masks' in self.edit_pool.keys()

        # Separate keys, values, masks
        self.keys = self.edit_pool['keys']
        self.values = self.edit_pool['values']
        self.masks = self.edit_pool['masks']
        self.labels = self.edit_pool['labels']

        # Create transforms
        transform = []
        if padding > 0:
            transform.append(transforms.Pad(padding, padding_mode='edge'))
        if len(transform) > 0:
            self.transform = transforms.Compose(transform)
        else:
            self.transform = None


    def __get_item__(self, index):
        cur_idxs = self.edit_idxs[index] # list of indices
        cur_keys = self.keys[cur_idxs]
        cur_values = self.values[cur_idxs]
        cur_labels = self.labels[cur_idxs]

        if self.masks is not None:
            cur_masks = self.masks[cur_idxs]
        else:
            masks_shape = list(cur_keys.shape)
            masks_shape[-3] = 1
            cur_masks =  torch.ones(masks_shape)

        cur_keys = self.transform(cur_keys)
        cur_values = self.transform(cur_values)
        cur_masks = self.transform(cur_masks)



        # This is the format EAC expects
        cur_edit_data = {
            'imgs': cur_values,
            'modified_imgs': cur_keys,
            'masks': cur_masks
        }

        return cur_edit_data, cur_labels