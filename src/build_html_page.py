import os, sys
import argparse
import datetime
import shutil
from airium import Airium

from utils import ensure_dir, read_lists

save_dir = os.path.join('html')
input_root_dir = os.path.join('saved', 'segmentation', 'semantics')
run_id = 'cat_20'
segmentation_save_dir = os.path.join(input_root_dir, run_id)
visualization_save_dir = os.path.join(save_dir, run_id, 'images')

def get_common_dir_path(paths):
    '''
    Get longest common directory path from all the paths
    '''
    common_dir_path = paths[0]
    # while common_dir_path not in paths[1]:
    #     common_dir_path = os.path.dirname(common_dir_path)
    for path in paths:
        while common_dir_path not in path:
            common_dir_path = os.path.dirname(common_dir_path)
        # assert

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
    # Obtain the common path for each input directory
    # common_dir_path = input_dirs[0]
    # while common_dir_path not in input_dirs[1]:
    #     common_dir_path = os.path.dirname(common_dir_path)
    # for input_dir in input_dirs:
    #     if common_dir_path not in input_dir:
    #         common_dir_path = os.path.dirname(common_dir_path)
    common_dir_path = get_common_dir_path(input_dirs)
    common_dir_path_len = len(common_dir_path)
    # print("common_dir_path: {}".format(common_dir_path))

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
    save_ids_paths = []
    for idx, input_dir in enumerate(input_dirs):
        id_ = input_dir[common_dir_path_len+1:]
        # print(id_)
        save_dir = os.path.join(output_dir, id_)
        ensure_dir(save_dir)
        # print(save_dir)

        for file_name in file_names:
            src_path = os.path.join(input_dir, file_name)
            dst_path = os.path.join(save_dir, file_name)
            if overwrite or not os.path.isfile(dst_path):
                print("copying \n\tfrom: {} \n\tto: {}".format(src_path, dst_path))
                shutil.copyfile(src_path, dst_path)
            save_ids_paths.append((id_, dst_path))

        save_dirs.append(save_dir)

    return save_dirs, save_ids_paths

def build_html(asset_ids_paths,
               html_save_path):
    '''
    Given paths to assets to embed, build HTML page

    Arg(s):
        asset_ids_paths : list[(str, str)]
            id : string of header
            path : path to asset

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
            for asset_id, asset_path in asset_ids_paths:
                # Create new header
                if asset_id != prev_id:
                    with air.h3():
                        air(asset_id)
                    prev_id = asset_id

                # Embed asset as image
                relative_asset_path = os.path.relpath(asset_path, os.path.dirname(html_save_path))
                print(relative_asset_path)
                air.img(src=relative_asset_path)


    # Turn Airium object to html string
    html_string = str(air)
    print(html_string)
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
    print(html_asset_paths[:5])
    # Build page
    html_string = build_html(
        asset_ids_paths=html_asset_paths[:5],
        html_save_path=html_save_path)

    # Ensure directory for html_save_path exists
    ensure_dir(os.path.dirname(html_save_path))
    with open(html_save_path, 'wb') as f:
        f.write(bytes(html_string, encoding='utf8'))


if __name__ == "__main__":
    value_image_paths_path = os.path.join('paths', 'edits', 'semantics', 'cat', '0124_142942', 'value_images_logits.txt')
    value_image_paths = read_lists(value_image_paths_path)
    input_dirs = [os.path.dirname(path) for path in value_image_paths]
    file_names = ['logits_cumulative_modifying.png', 'softmax_cumulative_modifying.png', 'target_logits_v_n_images.png', 'target_softmax_v_n_images.png']
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