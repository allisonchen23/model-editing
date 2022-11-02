import os, sys
import torch

import datasets
def run_model(dataset_path,
              model_restore_path,
              model_type,
              data_split='test',
              batch_size=128,
              normalize=False,
              mean=None,
              std=None,
              n_threads=8,
              device='cuda'):
    
    # Select device
    if device == 'cuda':
        device = torch.device('cuda')
    elif device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError('Unsupported device {}'.format(device))
        
    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        datasets.get_dataset(
            dataset_path=dataset_path,
            split=data_split,
            normalize=normalize,
            mean=mean,
            std=std),
        batch_size=batch_size,
        num_workers=n_threads,
        shuffle=False,
        drop_last=False)
    
    # Restore model