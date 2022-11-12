from torchvision import datasets, transforms
import os
from base import BaseDataLoader
from torch.utils.data import DataLoader


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
                 train=True,
                 normalize=True,
                 means=None,
                 stds=None,
                 num_workers=8):

        # Normalize data
        self.trsfm = [transforms.ToTensor()]
        if normalize:
            assert means is not None
            assert stds is not None
            self.trsfm.append(transforms.Normalize(mean=means, std=stds))

        if train:
            split_data_dir = os.path.join(data_dir, 'train')
        else:
            split_data_dir = os.path.join(data_dir, 'test')

        self.dataset = datasets.ImageFolder(
            root=split_data_dir,
            transform=transforms.Compose(self.trsfm))
        self.train = train
        self.n_samples = len(self.dataset)
        self.data_dir = data_dir
        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'drop_last': False
        }

        super().__init__(self.dataset, shuffle=shuffle, **self.init_kwargs)

    def split_validation(self):
        if not self.train:
            return None
        else:
            val_data_dir = os.path.join(self.data_dir, 'valid')
            val_dataset = datasets.ImageFolder(
                root=val_data_dir,
                transform=transforms.Compose(self.trsfm))
            return DataLoader(
                dataset=val_dataset,
                shuffle=False,
                **self.init_kwargs)
