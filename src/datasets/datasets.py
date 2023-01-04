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
                 normalize=True,
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

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.image_paths[index])
        image = load_image(image_path)

        label = int(self.labels[index])

        if self.return_paths:
            return image, label, image_path
        else:
            return image, label

    def __len__(self):
        return self.n_sample
