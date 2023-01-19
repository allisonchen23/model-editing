import numpy as np
import torch
import skimage.segmentation as segmentation
import skimage.filters as filters
import skimage.color as color
import sys

sys.path.insert(0, 'src')
import utils.visualizations as visualizations
from utils import informal_log

def segment(image, method, kwargs):
    '''
    Given an image and a segmentation method, and necessary keyword arguments, return segments

    Arg(s):
        image : C x H x W np.array
            Image to segment
        method : str
            type of segmentation method
        kwargs : dict
            keyword arguments for specific segmentation method

    Returns:
        segments : H x W np.array
            Each segment labeled by integer index
    '''

    # Assert image is in C x H x W shape
    assert len(image.shape) == 3, "Image expected to have shape C x H x W"
    assert image.shape[0] == 3, "Image expected to have 3 channels in dimension 0"
    kwargs['channel_axis'] = 0

    if method == 'felzenszwalb':
        segments = segmentation.felzenszwalb(
            image,
            **kwargs)
    elif method == 'quickshift':
        segments = segmentation.quickshift(
            image,
            **kwargs)
    elif method == 'slic':
        segments = segmentation.slic(
            image,
            **kwargs)

    elif method == 'watershed':
        del kwargs['channel_axis']
        gradient = filters.sobel(color.rgb2gray(np.transpose(image, axes=[1, 2, 0])))
        segments = segmentation.watershed(
            gradient,
            **kwargs)
    else:
        raise ValueError("Segmentation method {} not supported.".format(method))
    return segments

'''
Modification Methods
'''
def mask_out(image, mask):
    '''
    Given an image and binary mask, black out where mask is activated

    Arg(s):
        image : C x H x W np.array
            original image
        mask : C x H x W np.array (2D array broadcasted in C dimension)
            mask

    Returns:
        image : C x H x W np.array
            image with masked portion blacked out
    '''
    image = np.where(mask == 1, 0, image)
    return image

def gaussian_noise(image, mask, mean=0, std=0.1, seed=None):
    '''
    Given an image and binary mask, apply Gaussian noise to masked region

    Arg(s):
        image : C x H x W np.array
            original image
        mask : C x H x W np.array (2D array broadcasted in C dimension)
            mask
        mean : float
            mean for Gaussian distribution
        std : float
            standard deviation for Gaussian distribution
        seed : int or None
            set random seed for Gaussian noise

    Returns:
        image : C x H x W np.array
            image with noise added to masked portion
    '''
    # Set seed
    np.random.seed(seed)
    # Obtain normal distribution for noise
    normal_distribution = np.random.normal(mean, std, size=image.shape)
    # Add to image
    image = np.where(mask == 1, image + normal_distribution, image)
    # Clip if necessary
    image = np.clip(image, 0, 1)

    return image


def modify_segments(image,
                    segments,
                    method,
                    **kwargs):
    '''
    Given an image, the segmentation, and modification, return list of images with each segment modified

    Arg(s):
        image : C x H x W np.array (np.float)
            original image
        segments : H x W np.array (np.int)
            segmented image (e.g. output of utils.segmentation_utils.segment()
        method : str
            type of modification to perform

    Returns:
        modified_images : list[C x H x W np.array]
            list of modified images, one per segment
    '''
    # Obtain number of segments
    unique_segments = np.unique(segments)
    n_segments = len(unique_segments)
    broadcasted_segments = np.broadcast_to(segments, shape=(3, *segments.shape))
    modified_images = []

    # Determine modification method
    if method == 'gaussian_noise':
        modification_func = gaussian_noise
        assert 'mean' in kwargs and 'std' in kwargs, "Gaussian noise requires keyword arguments 'mean' and 'std'"

    elif method == 'mask':
        modification_func = mask_out
        kwargs = {}
    else:
        raise ValueError("Modification method {} not supported".format(method))

    # Modify each segment
    for label in unique_segments:
        label_mask = np.where(segments == label, 1, 0)

        modified_image = modification_func(image, label_mask, **kwargs)
        modified_images.append(modified_image)

    return modified_images

def segment_modify_multi_method(image, methods_params, seed, save_path=None):
    '''
    Given an image and segmentation method parameters, segment using each method and
    return modifications

    Arg(s):
        image : torch.tensor
            original image
        method_params : list[(str, dict)]
            list of pairs of name of segmentation method and dictionary containing parameters
        seed : int or None
            to make deterministic
        save_path : str or None
            optional save path for torch data

    Returns:
        dict : { str : any }
            return data
    '''
    segmentations = []
    gaussian_modified_images = []
    masked_modified_images = []

    for method, params in methods_params:
        segments = segment(
            image,
            method=method,
            kwargs=params)
        segmentations.append(segments)

        # Modify segments with noise
        gaussian_modified_images.append(modify_segments(
            image,
            segments=segments,
            method='gaussian_noise',
            mean=0,
            std=0.1,
            seed=seed))

        # Modify segments with masking
        masked_modified_images.append(modify_segments(
            image=image,
            segments=segments,
            method='mask'))

    save_data = {
        "original_image": image,  # np.array
        "segmentation_methods": methods_params,  # list[(str, dict)]
        "segmentations": segmentations,  # list[np.array]
        "gaussian_modified_images": gaussian_modified_images,  # list[list[np.array]]
        "masked_modified_images": masked_modified_images  # list[list[np.array]]
    }

    if save_path is not None:
        torch.save(save_data, save_path)
    return save_data


def predict_modified_segmentations(modified_segmentations,
                                   model,
                                   device,
                                   target_class_idx,
                                   log_path,
                                   visualization_save_path=None):
    # Unpack data
    image = modified_segmentations['original_image']
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).type(torch.FloatTensor).to(device)

    methods = modified_segmentations['segmentation_methods']
    all_gaussian_modified_images = modified_segmentations['gaussian_modified_images']
    all_masked_modified_images = modified_segmentations['masked_modified_images']

    # Create lists for storing logits
    all_gaussian_modified_logits = []
    all_masked_modified_logits = []

    # First pass original image through model
    with torch.no_grad():
        original_logits = model(image)
        original_logits = original_logits.cpu().numpy()

    # Data structures for keeping track of successful modifications
    success_modifications = []  # store images
    success_names = []  # store names of successful images

    # Pass images through model to get predictions
    for idx, (method, _) in enumerate(methods):
        # Index into current modified images
        gaussian_modified_images = all_gaussian_modified_images[idx]
        masked_modified_images = all_masked_modified_images[idx]
        n_modified = len(gaussian_modified_images)

        # Stack along batch dimension
        all_images = np.stack(gaussian_modified_images + masked_modified_images, axis=0)
        # Convert to tensor & switch to GPU
        all_images = torch.from_numpy(all_images).type(torch.FloatTensor)
        all_images = all_images.to(device)

        # Forward through model
        with torch.no_grad():
            all_logits = model(all_images)

        # Obtain indices where the prediction is the target
        all_predictions = torch.argmax(all_logits, dim=1)
        success_idxs = (all_predictions == target_class_idx).nonzero().squeeze(dim=-1)

        # Store images in respective list
        success_images = all_images[success_idxs]
        # If only one image, expand dims to be 1 x C x H x W
        if len(success_images.shape) == 3:
            success_images = torch.unsqueeze(success_images, dim=0)
        success_modifications.append(success_images)

        # Store names of each image in respective list
        if success_idxs.shape[0] != 0:
            informal_log("{} segmentation had {} sucessful modification(s)...".format(method, success_idxs.shape[0]), log_path)
            for idx in success_idxs:
                if idx < n_modified:
                    mod_type = 'gaussian'
                else:
                    mod_type = 'masked'
                name = "{}_{}_{}.png".format(method, mod_type, idx % n_modified)
                success_names.append(name)

                informal_log("  {}".format(name), log_path)

    # Concatenate into one tensor
    success_modifications = torch.cat(success_modifications, dim=0).cpu().numpy()
    print(success_modifications.shape)
    if success_modifications.shape[0] == 0:
        informal_log("No successful modifications produced...", log_path)
        return success_modifications, success_names

    # Visualize
    if visualization_save_path is not None:
        visualizations.show_image_rows(
            images=[list(success_modifications)],
            image_titles=[success_names],
            figure_title='Correctly predicted')

    return success_modifications, success_names


def visualize_segmentations(image,
                            methods,
                            segmentations,
                            gaussian_modified_images,
                            masked_modified_images,
                            save_path=None):
    visualization_grid = []
    row_labels = []
    max_row_len = 0

    # Iterate through all segmentation methods
    for idx, (method, _) in enumerate(methods):
        # Obtain current segmentation and overlay boundaries
        cur_segmentation = segmentations[idx]
        segmentation_overlay = segmentation.mark_boundaries(np.transpose(image, axes=[1, 2, 0]), cur_segmentation)

        # Obtain Gaussian and masked modifications
        gaussian_modified_row = gaussian_modified_images[idx]
        masked_modified_row = masked_modified_images[idx]

        # Top row will be for Gaussian
        row_labels.append(method + ' Gaussian')
        row_images = [segmentation_overlay] + gaussian_modified_row
        visualization_grid.append(row_images)

        # Bottom row will be for masked
        row_labels.append(method + ' Masked')
        row_images = [segmentation_overlay] + masked_modified_row
        visualization_grid.append(row_images)

        # Update max_row_len for display purposes
        if len(row_images) > max_row_len:
            max_row_len = len(row_images)

    # Pad extra columns with None
    for idx, row in enumerate(visualization_grid):
        n_pad = max_row_len - len(row)
        padding = [None for i in range(n_pad)]
        visualization_grid[idx] = row + padding

    # Sanity check
    for row in visualization_grid:
        assert len(row) == max_row_len

    # Save path
    # visualization_save_path = os.path.join(save_dir, 'segment-visualizations.png')
    visualizations.show_image_rows(
        images=visualization_grid,
        row_labels=row_labels,
        save_path=save_path)
    if save_path is not None:
        print("Saved visualization to {}".format(save_path))

def calculate_change(anchor, data, abs_val=True):
    '''
    Calculate change between all data points and anchor

    Arg(s):
        anchor : C-dim np.array
            one data point (e.g. originally predicted logits)
        data : N x C np.array
            multiple data points
        abs_val : boolean
            if True, return absolute values of differences

    Returns:
        deltas : N x C np.array
            array of same shape as data computing the change between all data and anchor
    '''

    deltas = data - np.broadcast_to(anchor, shape=data.shape)

    if abs_val:
        return np.abs(deltas)
    else:
        return deltas

def get_most_changed_idxs(anchor, data, target=None, verbose=False):
    '''
    Given data, return dictionary with indices of data that is
        1. most changed overall
        2. most changed in predicted class

    Arg(s):
        anchor : C-dim np.array or torch.tensor
            the original predictions
        data : N x C np.array or torch.tensor
            N is number of data points
            C is number of classes
        target : int
            index of desired class
        verbose : boolean
            If true, print out a lot of stuff

    Returns:
        dict[str] : int
    '''

    if torch.is_tensor(anchor):
        anchor = anchor.cpu().numpy()
    if torch.is_tensor(data):
        data = data.cpu().numpy()

    # Get the original prediction and its logits values
    original_prediction = np.argmax(anchor)
    original_prediction_logits = anchor[:, original_prediction]

    # Determine which modified index has the most change in the predicted class
    modified_predicted_logits = data[:, original_prediction]
    delta_predicted_logits = modified_predicted_logits - original_prediction_logits
    idx_predicted_directional = np.argmin(delta_predicted_logits)  # want index of most decrease
    idx_predicted = np.argmax(np.abs(delta_predicted_logits))

    if verbose:
        print("Original prediction is {} (logits: {})".format(original_prediction, original_prediction_logits))
        print("Logits for all modified images for prediction {}: {}".format(original_prediction, modified_predicted_logits))
        print("Idx for largest decrease in predicted class: {}".format(idx_predicted_directional))
        print("Idx for largest absolute change: {}".format(idx_predicted))

    # Determine which modified index has most change in logits overall
    change_logits = np.abs(data - np.broadcast_to(original_prediction_logits, shape=data.shape))
    sum_change_logits = np.sum(change_logits, axis=1)
    idx_overall = np.argmax(sum_change_logits)
    if verbose:
        print("Overall changes in logits for each modification: {}".format(sum_change_logits))

    if target is None:
        return {
            "predicted": idx_predicted,
            "predicted-directional": idx_predicted_directional,
            "overall": idx_overall
               }

    # Obtain logits for target class
    original_target_logits = anchor[:, target]
    modified_target_logits = data[:, target]

    delta_target_logits = modified_target_logits - original_target_logits
    # Determine idx with most positive change in target class and largest absolute value change in target class
    idx_target_directional = np.argmax(delta_target_logits)
    idx_target = np.argmax(np.abs(delta_target_logits))
    if verbose:
        print("Original logits for target class ({}): {}".format(target, original_target_logits))
        print("Logits for target class ({}): {}".format(target, modified_target_logits))
        print("Directional index: {} non-directional index: {}".format(idx_target_directional, idx_target))

    # Determine which modified idx has most change in target (increase) and predicted (decrease) (directional)
    delta_target_predicted_logits_directional = delta_target_logits - delta_predicted_logits
    idx_target_predicted_directional = np.argmax(delta_target_predicted_logits_directional)

    # Determine which modified idx has the most net change in target and predicted (non-directional)
    delta_abs_target_logits = np.abs(delta_target_logits)
    delta_target_predicted_logits = delta_predicted_logits + delta_target_logits
    idx_target_predicted = np.argmax(delta_target_predicted_logits)

    if verbose:
        print("\nChange in logits:")
        print("Target: {}".format(delta_target_logits))
        print("Pred: \t{}".format(delta_predicted_logits))
        print("Both: \t{}".format(delta_target_predicted_logits_directional))

    return {
        "predicted-absolute": idx_predicted,
        "predicted-directional": idx_predicted_directional,
        "target-absolute": idx_target,
        "target-directional": idx_target_directional,
        "target-predicted-absolute": idx_target_predicted,
        "target-predicted-directional": idx_target_predicted_directional,
        "overall": idx_overall,
    }

