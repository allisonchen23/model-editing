import os, sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
import torchvision.datasets.utils as dataset_utils

sys.path.insert(0, 'src')
from utils import ensure_dir
from utils.dataset_utils import get_color_dict

SUPPORTED_DATASET_TYPES = ['2_Spurious_MNIST', '3_Spurious_MNIST', '2_Rand_MNIST', '3_Rand_MNIST']

def color_grayscale_arr(arr: np.array,
                        color_idx: int,
                        data_format='CHW',
                        dtype='float32'):
    """
        Converts grayscale image to either red, green, or keep as white

        Arg(s):
            arr : np.array
                H x W array with black and white values
            color_idx : int
                color index (0=red; 1=green; 2=white)
            data_format : str
                CHW or CHW depending on which dimension to put the RGB channels
        Returns:
            arr : np.array
                H x W x C or C x H x W array
    """
    assert arr.ndim == 2
    # dtype = arr.dtype
    dtype = np.dtype(dtype)
    h, w = arr.shape
    if data_format == 'CHW':
        arr = np.reshape(arr, [1, h, w])
        if color_idx == 0:  # red
            arr = np.concatenate([
                arr,
                np.zeros((2, h, w))], axis=0, dtype=dtype)
        elif color_idx == 1:  # green
            arr = np.concatenate([
                np.zeros((1, h, w)),
                arr,
                np.zeros((1, h, w))], axis=0, dtype=dtype)
        elif color_idx == 2:  # white
            arr = np.concatenate([
                arr,
                arr,
                arr], axis=0, dtype=dtype)
        return arr
    elif data_format == 'CHW':
        arr = np.reshape(arr, [h, w, 1])
        if color_idx == 0:  # red
            arr = np.concatenate([
                arr,
                np.zeros((h, w, 2))], axis=2, dtype=dtype)
        elif color_idx == 1:  # green
            arr = np.concatenate([
                np.zeros((h, w, 1)),
                arr,
                np.zeros((h, w, 1))], axis=2, dtype=dtype)
        elif color_idx == 2:  # white
            arr = np.concatenate([
                arr,
                arr,
                arr], axis=2, dtype=dtype)
        return arr
    else:
        raise ValueError("data_format {} not recognized.".format(data_format))

def is_valid_dataset_type(dataset_type: str):
    return dataset_type in SUPPORTED_DATASET_TYPES

def assign_color(label: int,
                 n_colors: int,
                 correlation_type: str,
                 train=True,
                 n_classes=10):
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
            if label < n_classes // 2:
                color_idx = 0  # red
            else:
                color_idx = 1  # green
        elif n_colors == 3:
            if label < n_classes // 3:
                color_idx = 0
            elif label < 2 * (n_classes // 3):
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

# def get_color_dict(dataset_type, n_classes=10):
#     color_dict = {}
#     if dataset_type == '2_Spurious_MNIST':
#         for i in range(n_classes):
#             if i < n_classes // 2:
#                 color_dict[i] = 0
#             else:
#                 color_dict[i] = 1
#     else:
#         raise ValueError("Dataset type {} not supported for get_color_dict()".format(dataset_type))

#     return color_dict


def partition_hold_out_per_class(n_per_class,
             labels,
             colors=None):
    '''
    Given number of images per class and the data, separate into two partitions

    Arg(s):
        n_per_class : int
            number of samples per class
        n_classes : int
            number of classes/labels
        images : N x C x H x W np.array
            images
        labels : N np.array
            class labels of images
        colors : N np.array
            color idxs

    Returns:
        (np.array, np.array) : tuple of idxs for hold out and remaining respectively
    '''
    # Set variables
    n_classes = len(np.unique(labels))
    if colors is None:
        colors = np.zeros_like(labels)
    n_colors = len(np.unique(colors))
    print("{} classes and {} colors.".format(n_classes, n_colors))

    hold_out_idxs = []
    remaining_idxs = []
    for class_idx in range(n_classes):
        for color_idx in range(n_colors):
            sample_idxs = np.where(
                np.logical_and(labels == class_idx, colors==color_idx))[0]
            assert n_per_class < len(sample_idxs), \
                "N_per_class of {} is too high for class {} and color {}. Only {} samples in this class and color".format(
                n_per_class, class_idx, color_idx, len(sample_idxs))

            # Get samples for hold out
            hold_out_idxs.append(sample_idxs[:n_per_class])
            remaining_idxs.append(sample_idxs[n_per_class:])

    # Concatenate lists into numpy arrays
    hold_out_idxs = np.concatenate(hold_out_idxs, axis=0)
    remaining_idxs = np.concatenate(remaining_idxs, axis=0)

    return hold_out_idxs, remaining_idxs


def save_test_set_congruency(train_colors,
                             test_labels,
                             test_colors,
                             dataset_dir):
    '''
    Given the label -> color mapping of the training set, partition the test set indices
        based on whether they are congruent with training or not

    Arg(s):
        train_colors : dict{int : int}
            dictionary of length n_classes that maps the label -> color
        test_labels : 1D np.array
            labels of test set
        test_colors : 1D np.array
            colors of test set

    '''
    assert len(test_colors) == len(test_labels), \
        "Length of test colors ({}) does not match test labels ({})".format(
        len(test_colors), len(test_labels))

    congruent_idxs = []
    incongruent_idxs = []

    # Iterate through all samples
    for idx, (test_label, test_color) in enumerate(zip(test_labels, test_colors)):
        # Congruent if the test color matches the training color dictionary
        if train_colors[test_label] == test_color:
            congruent_idxs.append(idx)
        else:
            incongruent_idxs.append(idx)

    # Assert all samples are accounted for
    n_congruent = len(congruent_idxs)
    n_incongruent = len(incongruent_idxs)
    assert n_congruent + n_incongruent == len(test_labels), \
        "Length of congruent ({}) and incongruent ({}) test samples doesn't add up to test set size ({})".format(
        n_congruent,
        incongruent_idxs,
        len(test_labels))

    print("There are {} congruent samples and {} incongruent samples".format(n_congruent, n_incongruent))

    # Concatenate into arrays
    congruent_idxs = np.array(congruent_idxs)
    incongruent_idxs = np.array(incongruent_idxs)

    # Save numpy arrays
    congruent_idxs_path = os.path.join(dataset_dir, 'test_congruent_idxs.pt')
    incongruent_idxs_path = os.path.join(dataset_dir, 'test_incongruent_idxs.pt')
    torch.save(congruent_idxs, congruent_idxs_path)
    torch.save(incongruent_idxs, incongruent_idxs_path)
    print("Saved congruent test idxs to {} and incongruent test idxs to {}".format(
        congruent_idxs_path, incongruent_idxs_path))

def assign_and_color(im: Image,
                     label: int,
                     n_colors: int,
                     correlation_type: str,
                     n_classes: int,
                     data_format: str):

    # Normalize by 255.0 to make into floats
    im_array = np.array(im) / 255.0

    # Determine which color to assign number
    color_idx = assign_color(
        label=label,
        n_colors=n_colors,
        correlation_type=correlation_type,
        train=True,
        n_classes=n_classes
    )

    # Convert to RGB channel array
    colored_arr = color_grayscale_arr(
        arr=im_array,
        color_idx=color_idx,
        data_format=data_format)

    return colored_arr, color_idx

def alter_images_and_save(data: tuple,
                          n_colors: int,
                          correlation_type: str,
                          n_classes: int,
                          data_format: str,
                          dataset_type: str,
                          save_path: str,
                          save_congruency: bool=False,
                          dataset_dir :str=None,
                          hold_out_idxs: list=None):
    '''
    Given data of tuple(images, labels), alter the images and save them
    '''
    images = []
    labels = []
    colors = []

    hold_out_images = []
    hold_out_labels = []
    hold_out_colors = []

    if hold_out_idxs is None:
        hold_out_idxs = []
    n_hold_out_per_class = len(hold_out_idxs)
    hold_out_idxs = set(hold_out_idxs)
    assert len(hold_out_idxs) == n_hold_out_per_class

    for idx, (im, label) in enumerate(tqdm(data)):
        colored_arr, color_idx = assign_and_color(
            im=im,
            label=label,
            n_colors=n_colors,
            correlation_type=correlation_type,
            n_classes=n_classes,
            data_format=data_format)

        if idx in hold_out_idxs:
            hold_out_images.append(colored_arr)
            hold_out_labels.append(label)
            hold_out_colors.append(color_idx)
        else:
            # Append image and label
            images.append(colored_arr)
            labels.append(label)
            colors.append(color_idx)

    # Make into numpy arrays
    images = np.stack(images, axis=0)
    labels = np.array(labels)
    colors = np.array(colors)

    data = {
        "images": images,
        "labels": labels,
        "colors": colors
        }
    torch.save(data, save_path)
    split = os.path.basename(save_path).split('.pt')[0]
    print("Saved data for {} {} to {}".format(dataset_type, split, save_path))

    # Process hold out data
    if n_hold_out_per_class > 0:
        hold_out_images = np.stack(hold_out_images, axis=0)
        hold_out_labels = np.array(hold_out_labels)
        hold_out_colors = np.array(hold_out_colors)

        hold_out_data = {
            "images": hold_out_images,
            "labels": hold_out_labels,
            "colors": hold_out_colors,
        }
        hold_out_save_path = os.path.join(os.path.dirname(save_path), '{}_hold_out_{}.pt'.format(split, n_hold_out_per_class // n_classes))
        torch.save(hold_out_data, hold_out_save_path)
        print("Saved hold out data for {} {} to {}".format(dataset_type, split, hold_out_save_path))

    # Option to save idxs of congruent/incongruent test samples
    if save_congruency:
        assert dataset_dir is not None
        train_color_dict = get_color_dict(dataset_type=dataset_type)
        save_test_set_congruency(
            train_colors=train_color_dict,
            test_labels=labels,
            test_colors=colors,
            dataset_dir=dataset_dir)

def prepare_colored_mnist(root: str,
                          dataset_type: str,
                          n_classes=10,
                          seed: int=0,
                          data_format: str='CHW',
                          n_hold_out_per_class: int=0,
                          save_test_congruency: bool=False):
    '''

    Arg(s):
        root : str
            path to directory where dataset will be held
        dataset_type : str
            dataset name (2SpuriousMNIST, 2RandMNIST, 3RandMNIST, ...)
        seed : int
            seed to set randomness
        data_format : str
            'CHW' or 'HWC' specify which dimension to store channels
        hold_out : int
            number of samples per class to hold out of test set for editing
        save_test_congruency : bool
            whether or not to save congruent and incongruent idxs
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

    print("Preparing training data...")
    train_save_path = os.path.join(dataset_dir, 'training.pt')
    alter_images_and_save(
        data=train_mnist,
        n_colors=n_colors,
        correlation_type=correlation_type,
        n_classes=n_classes,
        data_format=data_format,
        dataset_type=dataset_type,
        save_path=train_save_path)

    print("Preparing testing data...")
    if n_hold_out_per_class > 0:
        # check if hold out data already exists, if it does, load it, otherwise create it
        hold_out_idxs_path = os.path.join(dataset_dir, 'test_hold_out_idxs_{}.pt'.format(n_hold_out_per_class))
        remaining_idxs_path = os.path.join(dataset_dir, 'test_remaining_idxs_{}.pt'.format(n_hold_out_per_class))
        if os.path.exists(hold_out_idxs_path):
            hold_out_idxs = torch.load(hold_out_idxs_path)
        else:
            # Obtain idxs for hold out and save them
            hold_out_idxs, remaining_idxs = partition_hold_out_per_class(
                n_per_class=n_hold_out_per_class,
                labels=test_mnist.targets,
                colors=None)

            torch.save(hold_out_idxs, hold_out_idxs_path)
            torch.save(remaining_idxs, remaining_idxs_path)
        assert len(hold_out_idxs) == n_classes * n_hold_out_per_class
    else:
        hold_out_idxs = None

    # Save test set (either full or with the holdout set)
    test_save_path = os.path.join(dataset_dir, 'test.pt')
    alter_images_and_save(
        data=test_mnist,
        n_colors=n_colors,
        correlation_type='Rand',
        n_classes=n_classes,
        data_format=data_format,
        dataset_type=dataset_type,
        save_path=test_save_path,
        save_congruency=save_test_congruency,
        dataset_dir=dataset_dir,
        hold_out_idxs=hold_out_idxs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make Colored MNIST datasets')
    parser.add_argument('-r', '--root', required=True, type=str,
        help='Path to data root directory')
    parser.add_argument('-d', '--dataset_type', required=True, type=str,
        help='String specifying how to color MNIST in format of \'N_X_D\' where N is number of colors; X is correlation type; D is name of Dataset')
    parser.add_argument('-l', '--n_classes', default=10, type=int,
        help='Number of labels in dataset, default=10')
    parser.add_argument('-s', '--seed', default=0, type=int,
        help='Seed for randomness, default=0')
    parser.add_argument('-f', '--data_format', default='CHW', type=str,
        help='Data format (CHW or HWC). Default=CHW')
    parser.add_argument('-c', '--save_test_congruency', default=False, action='store_true',
        help='Boolean of whether or not to store test idxs congruency')
    parser.add_argument('-o', '--n_hold_out_per_class', default=0, type=int,
        help='Number of images per class to hold out, default=0')

    args = parser.parse_args()

    prepare_colored_mnist(
        root=args.root,
        dataset_type=args.dataset_type,
        n_classes=args.n_classes,
        seed=args.seed,
        data_format=args.data_format,
        save_test_congruency=args.save_test_congruency,
        n_hold_out_per_class=args.n_hold_out_per_class
    )