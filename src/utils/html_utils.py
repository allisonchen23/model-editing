import os, sys
import argparse
import datetime
import shutil
from airium import Airium
import re

from utils import ensure_dir, read_lists

input_root_dir = os.path.join('saved', 'segmentation', 'semantics')
run_id = 'dog_20'
save_dir = os.path.join('html', run_id)
segmentation_save_dir = os.path.join(input_root_dir)
visualization_save_dir = os.path.join(save_dir, 'images')

def get_common_dir_path(paths):
    '''
    Get longest common directory path from all the paths
    '''
    common_dir_path = paths[0]
    for path in paths:
        while common_dir_path not in path:
            common_dir_path = os.path.dirname(common_dir_path)

    return common_dir_path

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
    for idx, input_dir in enumerate(input_dirs):
        id_ = input_dir[common_dir_path_len+1:]
        # print(id_)
        save_dir = os.path.join(output_dir, id_)
        ensure_dir(save_dir)
        # For each input directory, store associated file paths in a list
        file_save_paths = []
        for file_name in file_names:
            src_path = os.path.join(input_dir, file_name)
            dst_path = os.path.join(save_dir, file_name)
            if overwrite or not os.path.isfile(dst_path):
                shutil.copyfile(src_path, dst_path)

            file_save_paths.append(dst_path)
        save_paths.append(file_save_paths)
        save_ids.append(id_)
        save_dirs.append(save_dir)

    return save_dirs, save_paths, save_ids

def build_html(file_paths,
               html_save_path,
               id_regex='/+[a-z0-9_]*\-[a-z0-9_]*\-[a-z0-9_]*/.*/'):
    '''
    Given paths to assets to embed, build HTML page

    Arg(s):
        file_paths : list[str]
            paths to each asset (sorted to group assets together)
        html_save_path : str
            where the html file will be saved to
        id_regex : str
            Regular expression to extract ID

    Returns:
        html_string : str
            html as a string
    '''

    # Create Airium object
    air = Airium()

    air('<!DOCTYPE html>')
    with air.html(lang="pl"):
        # Set HTML header
        with air.head():
            air.meta(charset="utf-8")
            air.title(_t="Cumulative Image Visualization")

        # Set HTML body
        with air.body():
            prev_id = ""
            for path in file_paths:
                # asset_id = os.path.join(
                #     os.path.basename(os.path.dirname(path)),
                #     os.path.basename(path))
                asset_id = re.search(id_regex, path).group()
                # Remove the start and trailing backslashes
                asset_id = asset_id[1:-1]
                # Create new header
                if asset_id != prev_id:
                    with air.h3():
                        air(asset_id)
                    prev_id = asset_id

                # Embed asset as image
                relative_asset_path = os.path.relpath(path, os.path.dirname(html_save_path))
                air.img(src=relative_asset_path, height=350)
                air.p("\n\n")


    # Turn Airium object to html string
    html_string = str(air)
    return html_string

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