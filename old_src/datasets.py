
from torch.utils.data import DataLoader
import torchvision
import os
import torchvision.transforms as transforms
import pytorch_lightning as pl
# cinic_directory = '/path/to/cinic/directory'
# cinic_mean = [0.47889522, 0.47227842, 0.43047404]
# cinic_std = [0.24205776, 0.23828046, 0.25874835]
# cinic_train = torch.utils.data.DataLoader(
#     torchvision.datasets.ImageFolder(cinic_directory + '/train',
#     	transform=transforms.Compose([transforms.ToTensor(),
#         transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
#     batch_size=128, shuffle=True)


class CINIC10ImageNetDataset(pl.LightningDataModule):
    def __init__(self,
                 dataset_paths,
                 normalize=False,
                 mean=None,
                 std=None,
                 batch_size=128,
                 n_threads=8):
        super().__init__()

        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.n_threads = n_threads

        # Perform validity checks
        if self.normalize:
            assert self.mean is not None and len(self.mean) == 3
            assert self.std is not None and len(self.std) == 3

        if 'train' in dataset_paths.keys():
            self.train_data_path = dataset_paths['train']
        else:
            self.train_data_path = None

        if 'val' in dataset_paths.keys():
            self.val_data_path = dataset_paths['val']
        else:
            self.val_data_path = None

        if 'test' in dataset_paths.keys():
            self.test_data_path = dataset_paths['test']
        else:
            self.test_data_path = None

    def prepare_data(self):
        pass

    def setup(self, stage):
        transform_operations = [transforms.ToTensor()]
        transform_operations.append(transforms.Normalize(mean=self.mean, std=self.std))

        if stage == "fit":
            self.train_dataset = torchvision.datasets.ImageFolder(
                self.train_data_path,
                transform=transforms.Compose(transform_operations))
            self.val_dataset = torchvision.datasets.ImageFolder(
                self.val_data_path,
                transform=transforms.Compose(transform_operations))
        elif stage == "test":
            self.test_dataset = torchvision.datasets.ImageFolder(
                self.test_data_path,
                transform=transforms.Compose(transform_operations))
        elif stage == "all":
            self.train_dataset = torchvision.datasets.ImageFolder(
                self.train_data_path,
                transform=transforms.Compose(transform_operations))
            self.val_dataset = torchvision.datasets.ImageFolder(
                self.val_data_path,
                transform=transforms.Compose(transform_operations))
            self.test_dataset = torchvision.datasets.ImageFolder(
                self.test_data_path,
                transform=transforms.Compose(transform_operations))
        else:
            raise ValueError("Stage {} not recognized. Try 'train', 'test', or 'all'".format(stage))

    def train_dataloader(self):
        dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.n_threads,
                shuffle=True,
                drop_last=False)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_threads,
            shuffle=False,
            drop_last=False)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_threads,
            shuffle=False,
            drop_last=False)
        return dataloader

    def dataloader(self,
                   type,
                   batch_size,
                   n_threads):
        '''
        Return the training dataloader
        '''
        assert type in ["train", "val", "test"]
        transform_operations = [transforms.ToTensor()]
        transform_operations.append(transforms.Normalize(mean=self.mean, std=self.std))

        if type == 'train':
            assert self.train_data_path is not None
            dataloader = DataLoader(
                torchvision.datasets.ImageFolder(
                    self.train_data_path,
                    transform=transforms.Compose(transform_operations)),
                batch_size=batch_size,
                num_workers=n_threads,
                shuffle=True,
                drop_last=False)
        elif type == 'val':
            assert self.val_data_path is not None
            dataloader = DataLoader(
                torchvision.datasets.ImageFolder(
                    self.val_data_path,
                    transform=transforms.Compose(transform_operations)),
                batch_size=batch_size,
                num_workers=n_threads,
                shuffle=False,
                drop_last=False)
        else:  # type is 'test'
            assert self.test_data_path is not None
            dataloader = DataLoader(
                torchvision.datasets.ImageFolder(
                    self.test_data_path,
                    transform=transforms.Compose(transform_operations)),
                batch_size=batch_size,
                num_workers=n_threads,
                shuffle=False,
                drop_last=False)
        return dataloader


def get_dataset(dataset_path,
                split='train',
                normalize=False,
                mean=None,
                std=None):
    '''
    Return the respective torchvision.datasets dataset

    Arg(s):
        split : str
            choose from ['train', 'test', 'valid']
        normalize : bool
            whether to normalize or not
        mean : list[float] or None
            if normalize, must not be None.
            Mean values of dataset for RGB values
        std : list[float] or None
            if normalize, must not be None.
            Standard deviation values of dataset for RGB values

    Returns torchvision.dataset
    '''

    assert split in ['train', 'test', 'valid']
    dataset_path = os.path.join(dataset_path, split)
    transform_operations = [transforms.ToTensor()]
    if normalize:
        assert mean is not None and len(mean) == 3
        assert std is not None and len(std) == 3
        transform_operations.append(transforms.Normalize(mean=mean, std=std))

    return torchvision.datasets.ImageFolder(
        dataset_path,
        transform=transforms.Compose(transform_operations))

