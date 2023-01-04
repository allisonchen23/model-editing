import os
import sys

sys.path.insert(0, 'src')
import utils

# INPUTS
DATASET_NAME = 'cinic-10-imagenet'
# DATASET_NAME = 'cinic-10-imagenet-dummy'
DATA_DIR = os.path.join('data', DATASET_NAME)
SPLITS = ['train', 'test', 'valid']
CLASS_LIST_PATH = os.path.join('metadata', 'cinic-10', 'class_names.txt')

# OUTPUTS
PATHS_DIR = os.path.join('paths', 'datasets', DATASET_NAME)
PATHS_FILEPATH = os.path.join(PATHS_DIR, '{}_images.txt')
LABELS_FILEPATH = os.path.join(PATHS_DIR, '{}_labels.txt')

def write_paths():
    print("Recording paths for dataset '{}'".format(DATA_DIR))
    # Load class list
    class_list = utils.read_lists(CLASS_LIST_PATH)
    # Build class name -> idx dictionary
    class_name_to_idx = {}
    for idx, class_name in enumerate(class_list):
        class_name_to_idx[class_name] = idx

    # Create path directories if not already existing
    utils.ensure_dir(PATHS_DIR)

    # Iterate through each split
    for split in SPLITS:
        data_split_dir = os.path.join(DATA_DIR, split)
        image_filepaths_path = PATHS_FILEPATH.format(split)
        labels_path = LABELS_FILEPATH.format(split) # os.path.join(PATHS_FILEPATH, '{}_labels.txt'.format(split))
        data_paths = []
        labels = []

        for class_name in os.listdir(data_split_dir):
            # Append class name to path
            class_dir = os.path.join(data_split_dir, class_name)
            # Obtain label idx for this class
            label = str(class_name_to_idx[class_name])

            # Iterate through all images
            for img_name in os.listdir(class_dir):
                if not os.path.splitext(img_name)[1] == '.png':
                    continue
                # image_filepath = os.path.join(class_dir, img_name)
                image_save_filepath = os.path.join(split, class_name, img_name)

                # Append path and label to respective lists
                data_paths.append(image_save_filepath)
                labels.append(label)

        # Save lists
        utils.write_lists(image_filepaths_path, data_paths)
        utils.write_lists(labels_path, labels)

        print("Writing paths for {} split to:".format(split))
        print("\tImages: {}".format(image_filepaths_path))
        print("\tLabels: {}".format(labels_path))

if __name__ == "__main__":
    write_paths()