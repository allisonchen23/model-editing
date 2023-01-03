from torchvision import datasets, transforms
import os
from base import BaseDataLoader
from torch.utils.data import DataLoader
from torch import multiprocessing


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CINIC10DataLoader(DataLoader):
    '''
    CINIC10 DataLoader
    '''

    def __init__(self,
                 data_dir,
                 batch_size,
                 shuffle=True,
                 split='train',
                 normalize=True,
                 means=None,
                 stds=None,
                 num_workers=8,
                 return_paths=False):

        assert split in ['train', 'valid', 'test'], "Split must be in ['train', 'valid', 'test']"
        # Normalize data
        self.trsfm = [transforms.ToTensor()]
        if normalize:
            assert means is not None
            assert stds is not None
            self.trsfm.append(transforms.Normalize(mean=means, std=stds))

        # Obtain subdirectory for data
        self.split_data_dir = os.path.join(data_dir, split)
        # Obtain name of dataset
        self.data_name = os.path.basename(data_dir)

        # Create dataset
        # self.dataset = datasets.ImageFolder(
        #     root=self.split_data_dir,
        #     transform=transforms.Compose(self.trsfm))

        self.dataset = ImageFolderWithPaths(
            root=self.split_data_dir,
            transform=transforms.Compose(self.trsfm),
            return_paths=return_paths)
        self.return_paths = return_paths

        # Set variables
        self.n_samples = len(self.dataset)
        self.data_dir = data_dir
        # Set sharing strategy to avoid "too many files open"
        sharing_strategy = 'file_system'
        multiprocessing.set_sharing_strategy(sharing_strategy)
        def set_worker_sharing_strategy(worker_id):
            multiprocessing.set_sharing_strategy(sharing_strategy)
        print("num_workers: {}".format(num_workers))
        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'drop_last': False,
            'worker_init_fn': set_worker_sharing_strategy
        }
        # Create dataloader
        super().__init__(self.dataset, shuffle=shuffle, **self.init_kwargs)

    def get_data_dir(self):
        return self.split_data_dir

    def get_data_name(self):
        return self.data_name

    def get_return_paths(self):
        return self.return_paths

    def get_class_idx_maps(self):
        class_to_idx_map = self.dataset.class_to_idx
        idx_to_class_map = {}
        for class_name, idx in class_to_idx_map.items():
            idx_to_class_map.update({idx: class_name})
        return class_to_idx_map, idx_to_class_map

class ImageFolderWithPaths(datasets.ImageFolder):

    def __init__(self, root, transform, return_paths=False):
        '''
        Add variable to determine whether or not the dataloader should return path to data
        '''
        super().__init__(
            root=root,
            transform=transform)

        self.return_paths = return_paths

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # Return path to file if specified
        if self.return_paths:
            path = self.imgs[index][0]
            return (original_tuple + (path,))
        else:
            return original_tuple


