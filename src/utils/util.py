import json
import os, shutil
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def read_paths(filepath):
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


def write_paths(filepath, paths):
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


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


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
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def update_tensorboard(self, key, column='average'):
        if self.writer is not None:
            if column == 'total':
                value = self._data.total[key]
            elif column == 'counts':
                value = self._data.counts[key]
            elif column == 'average':
                value = self._data.average[key]
            else:
                raise Warning("Invalid column header {} for MetricTracker.update_tensorboard()".format(column))
                return
            self.writer.add_scalar(key, value)

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
