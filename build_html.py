# General imports
# import torch
import numpy as np
import os, sys
import json
from tqdm import tqdm
# import pandas as pd
from airium import Airium
import re
import argparse

# Local imports
sys.path.insert(0, 'src')
from utils import read_json, read_lists, ensure_dir
from utils.df_utils import load_and_preprocess_csv, get_sorted_idxs
from utils.html_utils import save_visualizations_separately, build_html
from utils.visualizations import bar_graph
# from parse_config import ConfigParser
# from data_loader import data_loaders
# import model.model as module_arch

def build_summary_page(results_root_dir,
                       html_save_dir,
                       relative_input_dirs,
                       file_names=None,
                       overwrite=False):
    '''
    Given the path to results directory,
        1) copy files to html_save_dir/assets
        2) build html page with summary of edits for this class

    Arg(s):
        results_root_dir : str
            where results_table.csv is stored
        html_save_dir : str
            where the assets/ directory should go and the html file
        relative_input_dir : list[str]
            relative path from results_root_dir where files are stored
        file_names : list[list[str]]
            list of file names inside of results_root_dir/relative_input_dir
            if None, copy all files
    '''
    # Create list of full input directories
    input_dirs = []
    for relative_input_dir in relative_input_dirs:
        input_dir = os.path.join(results_root_dir, relative_input_dir)
        if not os.path.isdir(input_dir):
            raise ValueError("Invalid directory '{}'".format(input_dir))
        input_dirs.append(input_dir)
    if file_names is None:
        file_names = []
        for input_dir in input_dirs:
            file_names.append(os.listdir(input_dir))
    else:
        assert len(file_names) == len(input_dirs), \
            "Not equal number of elements in file_names ({}) and input_dirs ({})".format(len(file_names), len(input_dirs))
        for group_idx, file_names_group in enumerate(file_names):
            if file_names_group is None:
                file_names[group_idx] = os.listdir(input_dirs[group_idx])

    assert len(file_names) == len(input_dirs)


    _, html_summary_save_paths, _ = save_visualizations_separately(
        input_dirs=[input_dir],
        file_names=file_names,
        output_dir=html_save_dir,
        overwrite=overwrite)


# copy_files(input

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', required=True, help='Path to config file')

    args = parser.parse_args()

    config = read_json(args.config)
    print(config['file_names'])

    build_summary_page(
        results_root_dir=config['results_root_dir'],
        html_save_dir=config['html_save_dir'],
        relative_input_dirs=config['relative_input_dirs'],
        file_names=config['file_names'],
        overwrite=config['overwrite'])

