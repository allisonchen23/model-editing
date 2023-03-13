import os
import torch

from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import datasets
import sys

sys.path.insert(0, 'src')
from utils import load_image

class CINIC10Dataset(Dataset):
    '''
    Dataset for CINIC10
    '''

    def __init__(self,
                 data_dir,
                 image_paths,
                 labels,
                 return_paths=True,
                 normalize=False,
                 means=None,
                 stds=None):

        self.data_dir = data_dir
        self.image_paths = image_paths
        self.labels = labels
        self.n_sample = len(image_paths)
        self.return_paths = return_paths

        # Transforms
        self.transforms = [transforms.ToTensor()]
        if normalize:
            assert means is not None and stds is not None
            self.transforms.append(transforms.Normalize(mean=means, std=stds))
        # PyTorch will already switch axes to C x H x W :')
        self.transforms = transforms.Compose(self.transforms)


    def __getitem__(self, index):
        # Obtain path, load image, apply transforms
        image_path = os.path.join(self.data_dir, self.image_paths[index])
        image = load_image(image_path, data_format="HWC")
        image = self.transforms(image)

        # Obtain label
        label = int(self.labels[index])

        # Return data
        if self.return_paths:
            return image, label, image_path
        else:
            return image, label


    def __len__(self):
        return self.n_sample

class ColoredMNIST(datasets.VisionDataset):
    """
    Colored MNIST dataset for testing

    Args:
        root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
        env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
        transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    """
    def __init__(self,
                 root: str,
                 dataset_type: str,
                 split: str,
                 transform=None,
                 target_transform=None):
        super(ColoredMNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)

        # Assert valid directory and split
        dataset_dir = os.path.join(root, dataset_type)
        assert os.isdir(dataset_dir), "Directory '{}' does not exist.".format(dataset_dir)
        if split not in ['training', 'test']:
            raise ValueError("Data split '{}' not supported. Choose from 'training' or 'test'".format(split))

        # Load images and labels
        data_path = os.path.join(dataset_dir, "{}.pt".format(split))
        self.images, self.labels = torch.load(data_path)

        assert len(self.images) == len(self.labels), "Images and labels have different number of samples ({} and {} respectively)".format(
            len(self.images), len(self.labels))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # Obtain image and label
        img = self.images[index]
        target = self.labels[index]

        # Apply transformations (if applicable)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)