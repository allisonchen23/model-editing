import os, sys
import torch

sys.path.insert(0, 'src')
from utils import ensure_dir, log as informal_log
import utils.segmentation_utils as segmentation_utils

def segment_modify_save(image,
                        image_name,
                        segmentation_methods_params,
                        seed,
                        save_dir_root,
                        log_path=None,
                        save_segmentations=False):
    # Make save directories/paths for data
    save_dir = os.path.join(save_dir_root, image_name)
    ensure_dir(save_dir)
    informal_log("Made save directory at {}".format(save_dir), log_path)

    if save_segmentations:
        segmentations_save_path = os.path.join(save_dir, 'segmentations.pth')
    else:
        segmentations_save_path = None
    image_save_dir = os.path.join(save_dir, 'modified_images')
    ensure_dir(image_save_dir)
    informal_log("Segmentations will be saved as .pth to {}".format(segmentations_save_path), log_path)
    informal_log("Successfully modified images will be saved in {}".format(image_save_dir), log_path)

    # Make save directory to write the paths
    save_paths_dir = os.path.join('paths', 'edits', image_name)
    ensure_dir(save_paths_dir)
    # segment image
    # modify each segment
    modified_segmentations = segmentation_utils.segment_modify_multi_method(
        image=image,
        methods_params=segmentation_methods_params,
        seed=seed,
        save_path=segmentations_save_path)
    if save_segmentations:
        informal_log("Saving segmentations to {}".format(segmentations_save_path))
    else:
        informal_log("Segmented and modified images...")
    # run modifications thru model to see which are successful
    # store paths and images of successful modifications