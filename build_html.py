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
from utils.html_utils import save_summary_page
from utils.visualizations import bar_graph
# from parse_config import ConfigParser
# from data_loader import data_loaders
# import model.model as module_arch


def save_html_page(config):
    '''
    Copy over necessary files to create an HTML page from config file

    Arg(s):
        config : dict
            config file as a dictionary
    '''
    page_type = config['page_type']
    if page_type == 'summary':
        # Format strings
        target_class_name = config['target_class_name']
        n_select = config['n_select']
        page_type = config['page_type']
        results_root_dir = config['results_root_dir'].format(target_class_name, n_select)
        html_save_dir = config['html_save_dir'].format(target_class_name, n_select, page_type)
        title = config['title'].format(target_class_name, n_select)

        save_summary_page(
            title=title,
            results_root_dir=results_root_dir,
            html_save_dir=html_save_dir,
            relative_input_dirs=config['relative_input_dirs'],
            file_names=config['file_names'],
            group_headers=config['group_headers'],
            overwrite=config['overwrite'])
    else:
        raise ValueError("Unsupported page_type '{}'".format(page_type))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', required=True, help='Path to config file')

    args = parser.parse_args()
    config = read_json(args.config)
    save_html_page(config)

