import os, sys
import argparse
import datetime
import shutil
from airium import Airium
import re
from tqdm import tqdm

sys.path.insert(0, 'src')
from utils import ensure_dir, read_lists, get_common_dir_path

input_root_dir = os.path.join('saved', 'segmentation', 'semantics')
run_id = 'dog_20'
save_dir = os.path.join('html', run_id)
segmentation_save_dir = os.path.join(input_root_dir)
visualization_save_dir = os.path.join(save_dir, 'images')

def copy_assets(src_root_dir,
                       relative_input_paths,
                       dst_root_dir,
                       overwrite=False):
    '''
    Copy elements in source_root_dir/relative_input_dirs to dest_root_dir/relative_input_dirs

    Arg(s):
        src_root_dir : str
            root of source files
        relative_input_paths : list[str]
            relative paths from src_root_dir to files to copy
        dst_root_dir : str
            root to store copied files
        overwrite : boolean
            whether or not to overwrite existing files
    '''
    # Ensure destination directory exists & create list to store paths
    ensure_dir(dst_root_dir)
    dst_paths = []

    # Iterate through all paths
    for relative_input_path in relative_input_paths:
        # Obtain source and destination paths
        src_path = os.path.join(src_root_dir, relative_input_path)
        if not os.path.isfile(src_path):
            continue
        dst_path = os.path.join(dst_root_dir, relative_input_path)

        # Copy file over
        if overwrite or not os.path.isfile(dst_path):
            shutil.copyfile(src_path, dst_path)

        # Add destination path to list
        dst_paths.append(dst_path)

    return dst_paths
def save_visualizations_separately(input_dirs,
                                   file_names,
                                   output_dir,
                                   overwrite=False):
    '''
    Given list of input directories, save the files specified in file_names to corresponding directories in output_dir

    Arg(s):
        input_dirs : list[str]
        file_names : list[str]
        output_dir : str

    Returns:
        None
    '''
    common_dir_path = get_common_dir_path(input_dirs)
    common_dir_path_len = len(common_dir_path)

    '''
    make list to save paths
    for each file directory in input dir,
        save_dir = output dir + (input_dir - common_dir path) = ID
        for each file in file names
            make complete path
            copy file to save_dir + (copy input_dir - common_dir path + filename)
        save the directory in list
    return list
    '''
    save_dirs = []
    save_ids = []
    save_paths = []
    for idx, input_dir in enumerate(tqdm(input_dirs)):
        id_ = input_dir[common_dir_path_len+1:]
        # print(id_)
        save_dir = os.path.join(output_dir, id_)
        ensure_dir(save_dir)
        # For each input directory, store associated file paths in a list
        file_save_paths = []
        for file_name in file_names:
            src_path = os.path.join(input_dir, file_name)
            # If src file doesn't exist, continue
            if not os.path.isfile(src_path):
                continue
            dst_path = os.path.join(save_dir, file_name)

            if overwrite or not os.path.isfile(dst_path):
                shutil.copyfile(src_path, dst_path)

            file_save_paths.append(dst_path)
        save_paths.append(file_save_paths)
        save_ids.append(id_)
        save_dirs.append(save_dir)

    return save_dirs, save_paths, save_ids

def build_html_summary(title,
               file_paths,
               headers,
               html_save_path,
               texts=None):
            #    id_regex='/+[a-z0-9_]*\-[a-z0-9_]*\-[a-z0-9_]*/.*/'):
    '''
    Given paths to assets to embed, build HTML page

    Arg(s):
        title : str
            title of HTML page
        file_paths : list[list[str]]
            paths to each asset (sorted to group assets together)
        headers : list[str]
            list of headers corresponding with each group of file_paths
        html_save_path : str
            where the html file will be saved to
        texts : list[list[str]] or None
            list of strings to display with each group
            inner list is different paragraphs/new lines
        id_regex : str
            Regular expression to extract ID

    Returns:
        html_string : str
            html as a string
    '''
    n_data = len(file_paths)
    # Create Airium object
    air = Airium()

    air('<!DOCTYPE html>')
    with air.html(lang="pl"):
        # Set HTML header
        with air.head():
            air.meta(charset="utf-8")
            air.title(_t=title)

        # Set HTML body
        # text_idx = 0
        with air.body():
            with air.h1():
                air(title)

            for group_idx, (header, group_file_paths) in enumerate(zip(headers, file_paths)):
                with air.h3():
                    air(header)

                # Display text
                if texts is not None:
                    for text in texts[group_idx]:
                        air.p(text)
                # Add assets as images
                for path in group_file_paths:
                    relative_asset_path = os.path.relpath(path, os.path.dirname(html_save_path))
                    air.img(src=relative_asset_path)
                    air.p("\n\n")
    # Turn Airium object to html string
    html_string = str(air)
    return html_string

def save_summary_page(title,
                       results_root_dir,
                       html_save_dir,
                       relative_input_dirs,
                       group_headers=None,
                       file_names=None,
                       overwrite=False):
    '''
    Given the path to results directory,
        1) copy files to html_save_dir/assets
        2) build html page with summary of edits for this class

    Arg(s):
        title : str
            Name for the HTML page
        results_root_dir : str
            where results_table.csv is stored
        html_save_dir : str
            where the assets/ directory should go and the html file
        relative_input_dirs : list[str]
            relative path from results_root_dir where files are stored
        group_headers : list[str]
            list of section headers in HTML file to split up the input directories
        file_names : list[list[str]]
            list of file names inside of results_root_dir/relative_input_dir
            if None, copy all files
        overwrite : bool
            whether to overwrite existing files when copying
    '''

    html_save_path = os.path.join(html_save_dir, 'summary.html')
    html_asset_save_dir = os.path.join(html_save_dir, 'assets')

    ensure_dir(html_asset_save_dir)
    n_groups = len(relative_input_dirs)
    if group_headers is None:
        group_headers = relative_input_dirs
    assert len(group_headers) == n_groups

    # If no file names provided, copy all files in all directories
    if file_names is None:
        file_names = []
        for input_dir in relative_input_dirs:
            files = os.listdir(input_dir)
            files.sort()
            file_names.append(files)
    else:
        # Check same length lists
        assert len(file_names) == len(relative_input_dirs), \
            "Not equal number of elements in file_names ({}) and input_dirs ({})".format(len(file_names), len(relative_input_dirs))
        # If any of the file_names lists are None, copy all files in that relative directory
        for group_idx, file_names_group in enumerate(file_names):
            if file_names_group is None:
                files = os.listdir(os.path.join(results_root_dir, relative_input_dirs[group_idx]))
                files.sort()
                file_names[group_idx] = files

    assert len(file_names) == len(relative_input_dirs)

    # Copy files from results_root_dir/input_dir/file_names to dst_root_dir with same relative paths
    asset_save_paths = []  # a list of lists
    for relative_input_dir, cur_file_names in zip(relative_input_dirs, file_names):
        src_root_dir = os.path.join(results_root_dir, relative_input_dir)
        save_paths = copy_assets(
            src_root_dir=src_root_dir,
            relative_input_paths=cur_file_names,
            dst_root_dir=html_asset_save_dir,
            overwrite=overwrite
        )
        asset_save_paths.append(save_paths)

    # Build HTML string
    html_string = build_html_summary(
        title=title,
        file_paths=asset_save_paths,
        headers=group_headers,
        html_save_path=html_save_path)

    # Write HTML file
    with open(html_save_path, 'wb') as f:
        f.write(bytes(html_string, encoding='utf-8'))
    print("Wrote summary HTML file to {}".format(html_save_path))

def create_html_visualization(input_dirs,
                              file_names,
                              html_asset_dir,
                              html_save_path,
                              overwrite=False):
                            #   local_paths):
    '''
    Given a list of paths of directories, create visualizations of the graphs and cumulative images
    '''

    # Copy desired assets
    html_asset_dirs, html_asset_paths = save_visualizations_separately(
        input_dirs=input_dirs,
        file_names=file_names,
        output_dir=html_asset_dir,
        overwrite=overwrite
    )
    print(len(html_asset_paths))
    # Build page
    html_string = build_html(
        asset_ids_paths=html_asset_paths,
        html_save_path=html_save_path)

    # Ensure directory for html_save_path exists
    ensure_dir(os.path.dirname(html_save_path))
    with open(html_save_path, 'wb') as f:
        f.write(bytes(html_string, encoding='utf8'))


if __name__ == "__main__":
    paths_dir = os.path.join('paths', 'edits', 'semantics', 'dog', '0124_160432')
    softmax_value_image_paths_path = os.path.join(paths_dir, 'value_images_softmax.txt')
    failure_value_image_paths_path = os.path.join(paths_dir, 'value_images_failures.txt')

    softmax_value_image_paths = read_lists(softmax_value_image_paths_path)
    failure_value_image_paths = read_lists(failure_value_image_paths_path)

    value_image_paths = softmax_value_image_paths + failure_value_image_paths
    input_dirs = [os.path.dirname(path) for path in value_image_paths]
    input_dirs = list(sorted(set(input_dirs)))
    # file_names = ['logits_cumulative_modifying.png', 'target_logits_v_n_images.png', 'softmax_cumulative_modifying.png',  'target_softmax_v_n_images.png']
    file_names = ['softmax_cumulative_modifying.png',  'target_softmax_v_n_images.png']

    html_save_path = os.path.join(save_dir, 'visualization.html')
    create_html_visualization(
        input_dirs=input_dirs,
        file_names=file_names,
        html_asset_dir=visualization_save_dir,
        html_save_path=html_save_path,
        overwrite=False)
    # print(input_dirs)
    # save_visualizations_separately(
    #     input_dirs=input_dirs,
    #     file_names=file_names,
    #     output_dir=visualization_save_dir
    # )