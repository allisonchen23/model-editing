import os
import shutil

DATA_DIR = 'data'
PARENT_DATASET_DIR = os.path.join(DATA_DIR, 'cinic-10-imagenet')
DUMMY_DATASET_DIR = PARENT_DATASET_DIR + '-dummy'
N_DIR_LEVELS = 3  # Number of directory steps to get to data
N_PER_CLASS = 100


def create_dummy_dataset():
    # def ignore_files(dir, files):
    #     return [f for f in files if os.path.isfile(os.path.join(dir, f))]

    # shutil.copytree(PARENT_DATASET_DIR, DUMMY_DATASET_DIR, ignore=ignore_files)
    def copy_n_files(path, n_files, cur_depth, max_depth):
        print(os.path.join(PARENT_DATASET_DIR, path))

        # Base case
        if cur_depth == max_depth:
            files = os.listdir(os.path.join(PARENT_DATASET_DIR, path))

            idx = 0
            while idx < n_files:
                filename = files[idx]
                filepath = os.path.join(PARENT_DATASET_DIR, path, filename)
                if not os.path.isfile(filepath):
                    continue
                if not os.path.isdir(os.path.join(DUMMY_DATASET_DIR, path)):
                    os.makedirs(os.path.join(DUMMY_DATASET_DIR, path))
                savepath = os.path.join(DUMMY_DATASET_DIR, path, filename)
                shutil.copy(filepath, savepath)
                idx += 1
        # Keep recursing through directories
        else:
            for dir in os.listdir(os.path.join(PARENT_DATASET_DIR, path)):
                if os.path.isdir(os.path.join(PARENT_DATASET_DIR, path, dir)):
                    copy_n_files(os.path.join(path, dir), n_files, cur_depth+1, max_depth)

    copy_n_files(
            path="",
            n_files=N_PER_CLASS,
            cur_depth=1,
            max_depth=N_DIR_LEVELS)


if __name__ == "__main__":
    create_dummy_dataset()
