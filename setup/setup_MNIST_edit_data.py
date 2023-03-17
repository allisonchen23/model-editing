import os, sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, 'src')
from utils import ensure_dir
from utils.visualizations import show_image_rows, make_grid
import model.metric as module_metric
from utils.dataset_utils import get_color_dict


def swap_color(image: np.array,
               data_format='CHW'):
    '''
    Given an RGB image, swap the R and G channels

    Arg(s):
        image : np.array
            C x H x W or H x W x C array representing image

    Returns:
        image :np.array
    '''
    swapped_image = np.copy(image)
    if data_format == 'CHW':
        assert image.shape[0] == 3
        swapped_image[[1, 0], :] = image[[0, 1], :]
    elif data_format == 'HWC':
        assert image.shape[2] == 3
        swapped_image[:, [1, 0]] = image[:, [0, 1]]
    else:
        raise ValueError("Data format {} not supported.".format(data_format))

    return swapped_image

def create_label_idx_dict(labels):
    '''
    Given list of labels, return dictionary that maps labels to all samples with that label

    Arg(s):
        labels : list[int]
            list of labels

    Returns:
        label_idx_dict : dict{int : list[int]}
            mapping label to indices with that label
    '''
    label_idx_dict = {}
    for idx, label in enumerate(labels):
        if label in label_idx_dict:
            label_idx_dict[label].append(idx)
        else:
            label_idx_dict[label] = [idx]
    return label_idx_dict


def make_edit_data(root: str,
                   dataset_type: str,
                   n_hold_out_per_class: int,
                   save_root: str,
                   n_classes: int=10):
    '''
    Given directory of hold out data, make the edit data for EAC

    Arg(s):
        root : str
            root to where dataset_type directory is stored (e.g. 'data')
        dataset_type : str
            type of MNIST dataset (e.g. 2_Spurious_MNIST)
        n_hold_out_per_class : int
            number of samples held out for each class
        save_root : str
            root to where to save edit data to (before dataset_type)
        n_classes : int
            number of classes

    '''
    # Paths to save data to
    save_edit_data_dir = os.path.join(save_root, dataset_type, 'hold_out_{}'.format(n_hold_out_per_class))
    ensure_dir(save_edit_data_dir)
    save_edit_data_path = os.path.join(save_edit_data_dir, 'test_hold_out_{}_eac.pt'.format(n_hold_out_per_class))
    save_label_idx_dict_path = os.path.join(save_edit_data_dir, 'label_idx_dict.pt')

    if os.path.exists(save_edit_data_path) and os.path.exists(save_label_idx_dict_path):
        print("Edit data already exists")
        return

    dataset_dir = os.path.join(root, dataset_type)

    # Load hold out data (modified same way test data is modified)
    hold_out_data_path = os.path.join(
        dataset_dir,
        'test_hold_out_{}.pt'.format(n_hold_out_per_class))
    hold_out_idxs_path = os.path.join(
        dataset_dir,
        'test_hold_out_idxs_{}.pt'.format(n_hold_out_per_class))
    assert os.path.exists(hold_out_data_path), "Path {} does not exist".format(hold_out_data_path)
    assert os.path.exists(hold_out_idxs_path), "Path {} does not exist".format(hold_out_idxs_path)

    hold_out_data = torch.load(hold_out_data_path)  # dictionary of np.arrays
    hold_out_idxs = torch.load(hold_out_idxs_path)

    hold_out_images = hold_out_data['images']
    hold_out_labels = hold_out_data['labels']
    hold_out_colors = hold_out_data['colors']

    # Obtain color dictionary
    color_dict = get_color_dict(
        dataset_type=dataset_type,
        n_classes=n_classes)

    # Arrays to store key and value images
    key_images = []  # store the image that is NOT congruent with training (so green 0-4s and red 5-9s)
    value_images = []  # store image that IS congruent with training (red 0-4s and green 5-9s)
    all_images = []

    for idx, (image, label, color) in enumerate(tqdm(zip(hold_out_images, hold_out_labels, hold_out_colors))):
        swapped_image = swap_color(image)

        if color_dict[label] == color: # if example is congruent with training, it is a value image
            key_image = swapped_image
            value_image = image
        else:
            key_image = image
            value_image = swapped_image
        all_images.append([key_image, value_image])
        # Append keys and values to lists
        key_images.append(key_image)
        value_images.append(value_image)

    # Turn list -> np.arrays
    key_images = np.stack(key_images, axis=0)
    value_images = np.stack(value_images, axis=0)

    label_idx_dict = create_label_idx_dict(labels=hold_out_labels)

    # Save edit data
    edit_data = {
        'keys': key_images,
        'values': value_images,
        'labels': hold_out_labels,
        'test_set_idxs': hold_out_idxs
    }
    torch.save(edit_data, save_edit_data_path)

    # Save label-idx dictionary
    torch.save(label_idx_dict, save_label_idx_dict_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make Colored MNIST edit data')
    parser.add_argument('-r', '--root', required=True, type=str,
        help='Path to data root directory')
    parser.add_argument('-d', '--dataset_type', required=True, type=str,
        help='String specifying how to color MNIST in format of \'N_X_D\' where N is number of colors; X is correlation type; D is name of Dataset')
    parser.add_argument('-s', '--save_root', required=True, type=str,
        help='Path to save edit data root directory')
    parser.add_argument('-o', '--n_hold_out_per_class', default=50, type=int,
        help='Number of images per class to hold out, default=50')
    parser.add_argument('-l', '--n_classes', default=10, type=int,
        help='Number of labels in dataset, default=10')

    args = parser.parse_args()
    make_edit_data(
        root=args.root,
        dataset_type=args.dataset_type,
        n_hold_out_per_class=args.n_hold_out_per_class,
        save_root=args.save_root,
        n_classes=args.n_classes)