import os, sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils

sys.path.insert(0, 'src')
from utils import ensure_dir

SUPPORTED_DATASET_TYPES = ['2_Spurious_MNIST', '3_Spurious_MNIST', '2_Rand_MNIST', '3_Rand_MNIST']

def color_grayscale_arr(arr: np.array,
                        color_idx: int,
                        data_format='CHW'):
    """
        Converts grayscale image to either red, green, or keep as white

        Arg(s):
            arr : np.array
                H x W array with black and white values
            color_idx : int
                color index (0=red; 1=green; 2=white)
            data_format : str
                CHW or HWC depending on which dimension to put the RGB channels
        Returns:
            arr : np.array
                H x W x C or C x H x W array
    """
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    if data_format == 'CHW':
        arr = np.reshape(arr, [1, h, w])
        if color_idx == 0:  # red
            arr = np.concatenate([
                arr,
                np.zeros((2, h, w), dtype=dtype)], axis=0)
        elif color_idx == 1:  # green
            arr = np.concatenate([
                np.zeros((1, h, w), dtype=dtype),
                arr,
                np.zeros((1, h, w), dtype=dtype)], axis=0)
        elif color_idx == 2:  # white
            arr = np.concatenate([
                arr,
                arr,
                arr], axis=0)
        return arr
    elif data_format == 'HWC':
        arr = np.reshape(arr, [h, w, 1])
        if color_idx == 0:  # red
            arr = np.concatenate([
                arr,
                np.zeros((h, w, 2), dtype=dtype)], axis=2)
        elif color_idx == 1:  # green
            arr = np.concatenate([
                np.zeros((h, w, 1), dtype=dtype),
                arr,
                np.zeros((h, w, 1), dtype=dtype)], axis=2)
        elif color_idx == 2:  # white
            arr = np.concatenate([
                arr,
                arr,
                arr], axis=2)
        return arr
    else:
        raise ValueError("data_format {} not recognized.".format(data_format))


# class ColoredMNIST(datasets.VisionDataset):
#     """
#     Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

#     Args:
#         root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
#         env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#         and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#         target and transforms it.
#     """
#     def __init__(self, root='./data', env='train1', transform=None, target_transform=None):
#         super(ColoredMNIST, self).__init__(root, transform=transform,
#                                     target_transform=target_transform)

#         self.prepare_colored_mnist()
#         if env in ['train1', 'train2', 'test']:
#             self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
#         elif env == 'all_train':
#             self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt')) + \
#                                 torch.load(os.path.join(self.root, 'ColoredMNIST', 'train2.pt'))
#         else:
#             raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         img, target = self.data_label_tuples[index]

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target

#     def __len__(self):
#         return len(self.data_label_tuples)

#     def prepare_colored_mnist(self):
#         colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
#         if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) \
#             and os.path.exists(os.path.join(colored_mnist_dir, 'train2.pt')) \
#             and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
#             print('Colored MNIST dataset already exists')
#             return

#         print('Preparing Colored MNIST')
#         train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

#         train1_set = []
#         train2_set = []
#         test_set = []
#         for idx, (im, label) in enumerate(train_mnist):
#             if idx % 10000 == 0:
#                 print(f'Converting image {idx}/{len(train_mnist)}')
#             im_array = np.array(im)

#             # Assign a binary label y to the image based on the digit
#             binary_label = 0 if label < 5 else 1

#             # Flip label with 25% probability
#             if np.random.uniform() < 0.25:
#                 binary_label = binary_label ^ 1

#             # Color the image either red or green according to its possibly flipped label
#             color_red = binary_label == 0

#             # Flip the color with a probability e that depends on the environment
#             if idx < 20000:
#                 # 20% in the first training environment
#                 if np.random.uniform() < 0.2:
#                     color_red = not color_red
#             elif idx < 40000:
#                 # 10% in the first training environment
#                 if np.random.uniform() < 0.1:
#                     color_red = not color_red
#             else:
#                 # 90% in the test environment
#                 if np.random.uniform() < 0.9:
#                     color_red = not color_red

#             colored_arr = color_grayscale_arr(im_array, red=color_red)

#             if idx < 20000:
#                 train1_set.append((Image.fromarray(colored_arr), binary_label))
#             elif idx < 40000:
#                 train2_set.append((Image.fromarray(colored_arr), binary_label))
#             else:
#                 test_set.append((Image.fromarray(colored_arr), binary_label))

#             # Debug
#             # print('original label', type(label), label)
#             # print('binary label', binary_label)
#             # print('assigned color', 'red' if color_red else 'green')
#             # plt.imshow(colored_arr)
#             # plt.show()
#             # break

#         dataset_utils.makedir_exist_ok(colored_mnist_dir)
#         torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
#         torch.save(train2_set, os.path.join(colored_mnist_dir, 'train2.pt'))
#         torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))



def is_valid_dataset_type(dataset_type: str):
    return dataset_type in SUPPORTED_DATASET_TYPES

def assign_color(label: int,
                 n_colors: int,
                 correlation_type: str,
                 train=True,
                 n_labels=10):
    '''
    Given a label, number of labels, and the correlation type, assign a color to this sample
    '''
    if correlation_type == 'Rand':
        bins = np.linspace(0, 1, num=n_colors+1)
        p = np.random.uniform()

        # Index of last bin low that p is greater than
        color_idx = np.asarray(bins < p).nonzero()[0][-1]
    elif correlation_type == 'Spurious':
        if n_colors == 2:
            if label < n_labels // 2:
                color_idx = 0
            else:
                color_idx = 1
        elif n_colors == 3:
            if label < n_labels // 3:
                color_idx = 0
            elif label < 2 * (n_labels // 3):
                color_idx = 1
            else:
                color_idx = 2
        else:
            raise ValueError("Spuriously colored only supported with 2 or 3 colors. Received {}".format(n_colors))
        # In test time, assign different spuriously correlated digits (essentially idx + 1)
        if not train:
            color_idx = (color_idx + 1) % n_colors
    else:
        raise ValueError("Only supports 'Spurious' and 'Rand' correlation type. Recieved {}".format(correlation_type))

    return color_idx


def prepare_colored_mnist(root: str,
                          dataset_type: str,
                          n_labels=10,
                          seed: int=0,
                          data_format: str='CHW'):
    '''

    Arg(s):
        root : str
            path to directory where dataset will be held
        dataset_type : str
            dataset name (2SpuriousMNIST, 2RandMNIST, 3RandMNIST, ...)
        seed : int
            seed to set randomness
    '''
    np.random.seed(seed)

    # Ensure valid dataset type and extract relevant info
    assert is_valid_dataset_type(dataset_type), "Invalid dataset type {}. Only the following are supported: {}".format(dataset_type, SUPPORTED_DATASET_TYPES)
    dataset_type_list = dataset_type.split('_')
    n_colors = int(dataset_type_list[0])
    correlation_type = dataset_type_list[1]

    dataset_dir = os.path.join(root, dataset_type)
    ensure_dir(dataset_dir)

    if os.path.exists(os.path.join(dataset_dir, 'training.pt')) \
        and os.path.exists(os.path.join(dataset_dir, 'test.pt')):
        print('Colored MNIST {} dataset already exists'.format(dataset_type))
        return

    print('Preparing Colored MNIST')
    train_mnist = datasets.mnist.MNIST(root, train=True, download=True)
    test_mnist = datasets.mnist.MNIST(root, train=False, download=True)

    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []
    color_counts = [0 for i in range(n_colors)]

    print("Preparing training data...")
    for idx, (im, label) in enumerate(tqdm(train_mnist)):
        # if idx % 10000 == 0:
        #     print(f'Converting image {idx}/{len(train_mnist)}')
        im_array = np.array(im)

        # Determine which color to assign number
        color_idx = assign_color(
            label=label,
            n_colors=n_colors,
            correlation_type=correlation_type,
            train=True,
            n_labels=n_labels
        )
        # Keep track of # samples for each color
        color_counts[color_idx] += 1

        # Convert to RGB channel array
        colored_arr = color_grayscale_arr(
            arr=im_array,
            color_idx=color_idx,
            data_format=data_format)
        # Append image and label
        train_imgs.append(colored_arr)
        train_labels.append(label)

    train_set = (train_imgs, train_labels)
    train_save_path = os.path.join(dataset_dir, 'training.pt')
    torch.save(train_set, train_save_path)
    print("Saved training data for {} to {}".format(dataset_type, train_save_path))

    print("Preparing testing data...")
    for idx, (im, label) in enumerate(tqdm(test_mnist)):
        # if idx % 10000 == 0:
        #     print(f'Converting image {idx}/{len(test_mnist)}')
        im_array = np.array(im)

        # Determine which color to assign number
        color_idx = assign_color(
            label=label,
            n_colors=n_colors,
            correlation_type=correlation_type,
            train=False,
            n_labels=n_labels
        )
        # Keep track of # samples for each color
        color_counts[color_idx] += 1

        # Convert to RGB channel array
        colored_arr = color_grayscale_arr(
            arr=im_array,
            color_idx=color_idx,
            data_format=data_format)
        # Append image and label
        test_imgs.append(colored_arr)
        test_labels.append(label)

    test_set = (test_imgs, test_labels)
    test_save_path = os.path.join(dataset_dir, 'test.pt')
    torch.save(test_set, test_save_path)
    print("Saved test data for {} to {}".format(dataset_type, test_save_path))

    # dataset_utils.makedir_exist_ok(colored_mnist_dir)
    # torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
    # torch.save(train2_set, os.path.join(colored_mnist_dir, 'train2.pt'))
    # torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make Colored MNIST datasets')
    parser.add_argument('-r', '--root', required=True, type=str,
        help='Path to data root directory')
    parser.add_argument('-d', '--dataset_type', required=True, type=str,
        help='String specifying how to color MNIST in format of \'N_X_D\' where N is number of colors; X is correlation type; D is name of Dataset')
    parser.add_argument('-l', '--n_labels', default=10, type=int,
        help='Number of labels in dataset, default=10')
    parser.add_argument('-s', '--seed', default=0, type=int,
        help='Seed for randomness, default=0')
    parser.add_argument('-f', '--data_format', default='CHW', type=str,
        help='Data format (CHW or HWC). Default=CHW')

    args = parser.parse_args()

    prepare_colored_mnist(
        root=args.root,
        dataset_type=args.dataset_type,
        n_labels=args.n_labels,
        seed=args.seed
    )