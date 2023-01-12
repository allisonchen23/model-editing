# General imports
import numpy as np
import os, sys
from datetime import datetime

# import skimage.segmentation as segmentation
# import skimage.filters as filters
# import skimage.color as color

# Local imports
sys.path.insert(0, 'src')
from utils import read_lists, write_lists, ensure_dir, list_to_dict, read_json, load_image
from utils.model_utils import prepare_device
from parse_config import ConfigParser
from segment import segment_modify_save

config_path = 'configs/copies/cinic10_imagenet_segmentation_edit_trials.json'
incorrect_images_path = 'metadata/CINIC10-ImageNet/dog/vgg16_bn/incorrect_image_paths.txt'
n_select = 20
seed = 0 # Set to None if want true randomness
timestamp = datetime.now().strftime(r'%m%d_%H%M%S')

def main():
    ### Obtain incorrect images to use for edit
    all_incorrect_images = np.array(read_lists(incorrect_images_path))
    n_images = len(all_incorrect_images)
    np.random.seed(seed)
    key_images = all_incorrect_images[np.random.randint(n_images, size=n_select)]
    print(key_images)

    # Obtain class list
    class_list_path = os.path.join('metadata', 'cinic-10', 'class_names.txt')
    class_list = read_lists(class_list_path)
    class_to_idx_dict = list_to_dict(class_list)

    # General save directories
    save_dir_root = os.path.join('saved', 'segmentations')
    if not os.path.isdir(save_dir_root):
        os.makedirs(save_dir_root)


    # Set parameters for segmentation
    felzenszwalb_params = {
        'scale': 0.9,
        'sigma': 0.25,
        'min_size': 50
    }
    quickshift_params = {
        'max_dist': 25,
        'kernel_size': 3,
        'sigma': 0.9,
    }
    slic_params = {
        'n_segments': 10,
    }

    watershed_params = {
        'markers': 10,
        'watershed_line': True
    }

    # Set segmentation methods
    methods = []
    methods.append(('felzenszwalb', felzenszwalb_params))
    methods.append(('quickshift', quickshift_params))
    methods.append(('slic', slic_params))
    methods.append(('watershed', watershed_params))

    # Load config file, and class names file
    config_json = read_json(config_path)
    config = ConfigParser(config_json, make_dirs=False)
    # class_names = read_lists(class_list_path)

    original_image_paths = {}
    modified_image_paths = {}
    for image_path in key_images:
        # Extract information for this image
        split = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
        class_name = os.path.basename(os.path.dirname(image_path))
        file_name = os.path.basename(image_path).split(".")[0]
        image_name = "{}-{}-{}".format(class_name, split, file_name)

        # Load image
        image = load_image(image_path, data_format='CHW')

        # Obtain target class
        target_class_idx = class_to_idx_dict[class_name] # 5 = dog

        cur_modified_image_paths = (segment_modify_save(
            image=image,
            image_name=image_name,
            segmentation_methods_params=methods,
            seed=seed,
            save_dir_root=save_dir_root,
            config=config,
            target_class_idx=target_class_idx,
            log_path=None,
            save_segmentations=True,
            save_visualizations=True))

        cur_original_image_paths = [image_path for i in range(len(cur_modified_image_paths))]
        assert len(cur_modified_image_paths) == len(cur_original_image_paths)

        # Add to master dictionary (separated by class)
        if class_name in original_image_paths:
            original_image_paths[class_name] += cur_original_image_paths
            modified_image_paths[class_name] += cur_modified_image_paths
        else:
            original_image_paths[class_name] = cur_original_image_paths
            modified_image_paths[class_name] = cur_modified_image_paths

    # Save image paths
    n_modified_total = 0
    for class_name in original_image_paths.keys():
        save_paths_dir = os.path.join('paths', 'edits', class_name, timestamp)
        ensure_dir(save_paths_dir)

        save_original_paths_path = os.path.join(save_paths_dir, 'key_images.txt')
        save_modified_paths_path = os.path.join(save_paths_dir, 'value_images.txt')

        write_lists(save_original_paths_path, original_image_paths[class_name])
        write_lists(save_modified_paths_path, modified_image_paths[class_name])

        n_modified_total += len(original_image_paths[class_name])
    print("Paths to images are saved as .txt files in {}".format(save_paths_dir))
    print("Saved total of {} edit image pairs".format(n_modified_total))

if __name__ == "__main__":
    main()