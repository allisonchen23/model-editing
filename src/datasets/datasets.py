import os
from torchvision import transforms
from torch.utils.data import Dataset
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
