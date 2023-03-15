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
    elif data_format == 'CHW':
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

def get_color_dict(dataset_type, n_classes=10):
    color_dict = {}
    if dataset_type == '2_Spurious_MNIST':
        for i in range(n_classes):
            if i < n_classes // 2:
                color_dict[i] = 0
            else:
                color_dict[i] = 1
    else:
        raise ValueError("Dataset type {} not supported for get_color_dict()".format(dataset_type))

    return color_dict


def hold_out(n_per_class,
             # n_classes,
             images,
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
    '''

    hold_out_images = []
    hold_out_labels = []
    hold_out_colors = []

    remaining_images = []
    remaining_labels = []
    remaining_colors = []

    n_classes = len(np.unique(labels))
    if colors is None:
        colors = np.zeros_like(labels)
    n_colors = len(np.unique(colors))
    print("{} classes and {} colors.".format(n_classes, n_colors))


    for class_idx in range(n_classes):
        for color_idx in range(n_colors):
            sample_idxs = np.where(
                np.logical_and(labels == class_idx, colors==color_idx))[0]
            assert n_per_class < len(sample_idxs), \
                "N_per_class of {} is too high for class {} and color {}. Only {} samples in this class and color".format(
                n_per_class, class_idx, color_idx, len(sample_idxs))

            # Get samples for hold out
            cur_hold_images = images[sample_idxs][:n_per_class]
            cur_hold_labels = labels[sample_idxs][:n_per_class]
            cur_hold_colors = colors[sample_idxs][:n_per_class]

            hold_out_images.append(cur_hold_images)
            hold_out_labels.append(cur_hold_labels)
            hold_out_colors.append(cur_hold_colors)

            # Get remaining samples
            cur_remaining_images = images[sample_idxs][n_per_class:]
            cur_remaining_labels = labels[sample_idxs][n_per_class:]
            cur_remaining_colors = colors[sample_idxs][n_per_class:]

            remaining_images.append(cur_remaining_images)
            remaining_labels.append(cur_remaining_labels)
            remaining_colors.append(cur_remaining_colors)


    hold_out_images = np.concatenate(hold_out_images, axis=0)
    hold_out_labels = np.concatenate(hold_out_labels, axis=0)
    hold_out_colors = np.concatenate(hold_out_colors, axis=0)

    remaining_images = np.concatenate(remaining_images, axis=0)
    remaining_labels = np.concatenate(remaining_labels, axis=0)
    remaining_colors = np.concatenate(remaining_colors, axis=0)

    hold_out_data = {
        'images': hold_out_images,
        'labels': hold_out_labels,
        'colors': hold_out_colors
    }

    remaining_data = {
        'images': remaining_images,
        'labels': remaining_labels,
        'colors': remaining_colors
    }

    return hold_out_data, remaining_data


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
    for idx, (test_label, test_color) in enumerate(zip(test_labels, test_colors)):
        if train_colors[test_label] == test_color:
            congruent_idxs.append(idx)
        else:
            incongruent_idxs.append(idx)

    n_congruent = len(congruent_idxs)
    n_incongruent = len(incongruent_idxs)
    assert n_congruent + n_incongruent == len(test_labels), \
        "Length of congruent ({}) and incongruent ({}) test samples doesn't add up to test set size ({})".format(
        n_congruent,
        incongruent_idxs,
        len(test_labels))

    print("There are {} congruent samples and {} incongruent samples".format(n_congruent, n_incongruent))

    congruent_idxs = np.array(congruent_idxs)
    incongruent_idxs = np.array(incongruent_idxs)

    # ensure_dir(dataset_dir)

    congruent_idxs_path = os.path.join(dataset_dir, 'test_congruent_idxs.pt')
    incongruent_idxs_path = os.path.join(dataset_dir, 'test_incongruent_idxs.pt')

    torch.save(congruent_idxs, congruent_idxs_path)
    torch.save(incongruent_idxs, incongruent_idxs_path)

    print("Saved congruent test idxs to {} and incongruent test idxs to {}".format(
        congruent_idxs_path, incongruent_idxs_path))

def prepare_colored_mnist(root: str,
                          dataset_type: str,
                          n_labels=10,
                          seed: int=0,
                          data_format: str='CHW',
                          edit_hold_out: int=0,
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
        edit_hold_out : int
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

    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []
    color_counts = [0 for i in range(n_colors)]
    train_color_idxs = []
    test_color_idxs = []

    print("Preparing training data...")
    for idx, (im, label) in enumerate(tqdm(train_mnist)):
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
        train_color_idxs.append(color_idx)

    # Make into numpy arrays
    train_imgs = np.stack(train_imgs, axis=0)
    train_labels = np.array(train_labels)
    train_color_idxs = np.array(train_color_idxs)

    train_set = {
        "images": train_imgs,
        "labels": train_labels,
        "colors": train_color_idxs
        }
    train_save_path = os.path.join(dataset_dir, 'training.pt')
    torch.save(train_set, train_save_path)
    print("Saved training data for {} to {}".format(dataset_type, train_save_path))

    print("Preparing testing data...")
    if edit_hold_out > 0:
        # check if hold out data already exists, if it does, load it, otherwise create it
        processed_mnist_dir = os.path.join(root, 'MNIST', 'processed')
        hold_out_path = os.path.join(processed_mnist_dir, 'test_hold_out.pt')
        remaining_path = os.path.join(processed_mnist_dir, 'test_remaining.pt')
        if os.path.exists(hold_out_path) and \
            os.path.exists(remaining_path):
            hold_out_mnist = torch.load(hold_out_path)
            remaining_mnist = torch.load(remaining_path)

        else:
            hold_out_dict, remaining_dict = hold_out(
                n_per_class=edit_hold_out,
                images=test_mnist[0].cpu().numpy(),
                labels=test_mnist[1].cpu().numpy(),
                colors=None)

            hold_out_mnist = (hold_out_dict['images'], hold_out_dict['labels'])
            remaining_mnist = (remaining_dict['images'], remaining_dict['labels'])

        assert len(hold_out_mnist[0]) == len(hold_out_mnist[1]), \
            "Length of hold out images ({}) does not match labels ({})".format(
                len(hold_out_mnist[0]), len(hold_out_mnist[1])
            )
        assert len(remaining_mnist[0]) == len(remaining_mnist[1]), \
            "Length of remaining test images ({}) does not match labels ({})".format(
                len(remaining_mnist[0]), len(remaining_mnist[1])
            )

        # TODO: iterate through hold out set to convert colors
        # Save hold out set
        # iterate through remaining_test to convert colors
        # Refactor everything that's in the for loop to a separate function to make this function smaller
    for idx, (im, label) in enumerate(tqdm(test_mnist)):
        # if idx % 10000 == 0:
        #     print(f'Converting image {idx}/{len(test_mnist)}')
        im_array = np.array(im)

        # Determine which color to assign number
        color_idx = assign_color(
            label=label,
            n_colors=n_colors,
            correlation_type='Rand',
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
        # Append image and label and color_idx
        test_imgs.append(colored_arr)
        test_labels.append(label)
        test_color_idxs.append(color_idx)

    # Make into numpy arrays
    test_imgs = np.stack(test_imgs, axis=0)
    test_labels = np.array(test_labels)
    test_color_idxs = np.array(test_color_idxs)

    # if edit_hold_out > 0:
    #     test_data = hold_out()


    test_set = {
        "images": test_imgs,
        "labels": test_labels,
        "colors": test_color_idxs
    }
    test_save_path = os.path.join(dataset_dir, 'test.pt')
    torch.save(test_set, test_save_path)
    print("Saved test data for {} to {}".format(dataset_type, test_save_path))

    if save_test_congruency:
        train_color_dict = get_color_dict(dataset_type=dataset_type)
        save_test_set_congruency(
            train_colors=train_color_dict,
            test_labels=test_labels,
            test_colors=test_color_idxs,
            dataset_dir=dataset_dir)


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
    parser.add_argument('-c', '--save_test_congruency', default=False, action='store_true',
        help='Boolean of whether or not to store test idxs congruency')

    args = parser.parse_args()

    prepare_colored_mnist(
        root=args.root,
        dataset_type=args.dataset_type,
        n_labels=args.n_labels,
        seed=args.seed,
        data_format=args.data_format,
        save_test_congruency=args.save_test_congruency
    )