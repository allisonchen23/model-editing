import os, sys
import torch
import skimage.segmentation as segmentation

sys.path.insert(0, 'src')
from utils import ensure_dir, informal_log, save_image
import utils.segmentation_utils as segmentation_utils
from utils.model_utils import prepare_device
import model.model as module_arch


def segment_modify_save(image,
                        image_name,
                        segmentation_methods_params,
                        seed,
                        save_dir_root,
                        config,
                        target_class_idx,
                        log_path=None,
                        save_segmentations=False,
                        save_visualizations=False):
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
    # save_paths_dir = os.path.join('paths', 'edits', image_name)
    # ensure_dir(save_paths_dir)
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

    # Save visualizations if desired
    if save_visualizations:
        visualization_save_path = os.path.join(save_dir, 'segmentation_visualizations.png')
        segmentation_utils.visualize_segmentations(
            image=image,
            methods=segmentation_methods_params,
            segmentations=modified_segmentations['segmentations'],
            gaussian_modified_images=modified_segmentations['gaussian_modified_images'],
            masked_modified_images=modified_segmentations['masked_modified_images'],
            save_path=visualization_save_path)

    # Load model
    layernum = config.config['layernum']
    device, _ = prepare_device(config['n_gpu'])
    model = config.init_obj('arch', module_arch, layernum=layernum)
    model.eval()

    if save_visualizations:
        visualization_save_path = os.path.join(save_dir, 'modified_segment_visualization.png')
    else:
        visualization_save_path = None

    # Obtain predictions for modified segmentations and store successful ones
    success_modifications, success_names = segmentation_utils.predict_modified_segmentations(
        modified_segmentations=modified_segmentations,
        model=model,
        device=device,
        target_class_idx=target_class_idx,
        log_path=None,
        visualization_save_path=visualization_save_path)

    save_paths = []
    # Save images to file system and store path to each image
    for success_name, success_image in zip(success_names, success_modifications):
        save_path = os.path.join(image_save_dir, success_name)
        save_paths.append(save_path)
        save_image(success_image, save_path)
        informal_log("Saving image to {}".format(save_path))

    return save_paths

    # store paths and images of successful modifications

