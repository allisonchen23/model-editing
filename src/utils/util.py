import json
import os, shutil
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np
from PIL import Image
import pickle

def read_lists(filepath):
    '''
    Stores a depth map into an image (16 bit PNG)
    Arg(s):
        path : str
            path to file where data will be stored
    '''

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip('\n')
            # If there was nothing to read
            if path == '':
                break
            path_list.append(path)

    return path_list


def write_lists(filepath, paths):
    '''
    Stores line delimited paths into file
    Arg(s):
        filepath : str
            path to file to save paths
        paths : list[str]
            paths to write into file
    '''

    with open(filepath, 'w') as o:
        for idx in range(len(paths)):
            o.write(paths[idx] + '\n')

def append_lists(filepath, paths):
    '''
    Stores line delimited paths into file
    Arg(s):
        filepath : str
            path to file to save paths
        paths : list[str]
            paths to write into file
    '''

    with open(filepath, 'w+') as o:
        for idx in range(len(paths)):
            o.write(paths[idx] + '\n')

def read_pickle(filepath):
    '''
    Return unserialized pickle object at filepath
    Arg(s):
        filepath : str
            path to pickle file

    Returns:
        object
    '''

    with open(filepath, 'rb') as f:
        return pickle.load(f)

def write_pickle(filepath, object):
    '''
    Serialize object as pickle file at filepath
    Arg(s):
        filepath : str
            file to write to
        object : any
            Serializable object
    '''
    # Create directory if it doesn't exist
    if not os.path.isdir(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    with open(filepath, 'wb') as f:
        pickle.dump(object, f)

def load_image(image_path, data_format='HWC', resize=None):
    '''
    Load image and return as CHW np.array

    Arg(s):
        image_path : str
            path to find image
        data_format : str
            order of channels to return
        resize : tuple(int, int) or None
            the resized shape (H, W) or None

    Returns :
        C x H x W np.array normalized between [0, 1]
    '''
    image = Image.open(image_path).convert("RGB")
    if resize is not None:
        image = image.resize(resize)
    # Convert to numpy array
    image = np.asarray(image, float)

    # Normalize between [0, 1]
    image = image / 255.0

    if data_format == "HWC":
        return image.astype(np.float32)
    elif data_format == "CHW":
        # Make channels C x H x W
        image = np.transpose(image, (2, 0, 1))
        return image.astype(np.float32)
    else:
        raise ValueError("Unsupported data format {}".format(data_format))

def save_image(image, save_path):
    '''
    Given the image, save as PNG to save_path

    Arg(s):
        image : np.array
            image to save
        save_path : str
            location in file system to save to

    Returns:
        None
    '''
    # Create save directory if it doesn't already exist
    ensure_dir(os.path.dirname(save_path))

    # Convert to integer values
    if image.dtype == np.float32:
        image = image * 255.0
        image = image.astype(np.uint8)
    # Transpose if in format of C x H x W to H x W x C
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    # Convert array -> image and save
    image = Image.fromarray(image)
    image.save(save_path)

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def copy_file(file_path, save_dir):
    '''
    Copy the file from file_path to the directory save_dir
    Arg(s):
        file_path : str
            file to copy
        save_dir : str
            directory to save file to
    Returns : None
    '''
    # Assert files/directories exist
    assert os.path.exists(file_path)
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, os.path.basename(file_path))
    shutil.copy(file_path, save_path)

def list_to_dict(list_):
    '''
    Given a list, return a dictionary keyed by the elements of the list to corresponding indices

    Arg(s):
        list_ : list[any]
            input list

    Returns:
        dict_: dict{ int : any}
            corresponding dictionary to list_
    '''
    dict_ = {}
    for idx, element in enumerate(list_):
        dict_[element] = idx

    return dict_

def print_dict(dictionary, indent_level=0):
    tabs = ""
    for i in range(indent_level):
        tabs += "\t"
    for key, val in dictionary.items():
        if type(val) == dict:
            print("{}{}".format(tabs, key))
            print_dict(val, indent_level=indent_level+1)
        else:

            print("{}{} : {}".format(tabs, key, val))


def informal_log(s, filepath=None, to_console=True):
    '''
    Logs a string to either file or console
    Arg(s):
        s : str
            string to log
        filepath
            output filepath for logging
        to_console : bool
            log to console
    '''

    if to_console:
        print(s)

    if filepath is not None:
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
            with open(filepath, 'w+') as o:
               o.write(s + '\n')
        else:
            with open(filepath, 'a+') as o:
                o.write(s + '\n')


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
