# model-editing
Code for model editing project

## Setup

### Data

#### CINIC-10

1. Download using `wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz`
2. Make a directory and move the `.tar.gz` file into directory: `mkdir cinic-10 && mv CINIC-10.tar.gz cinic-10/`
3. Unzip the tar file: `cd cinic-10 && tar -xvf CINIC-10.tar.gz`
4. In repository root, create a data folder and symlink the data: `mkdir data && ln -s </path/to/cinic-10> data/`

### Code

The directory `external_code` has code for

1. CINIC-10 from [here](https://github.com/BayesWatch/cinic-10)
2. CIFAR-10 pretrained models from [here](https://github.com/huyvnphan/PyTorch_CIFAR10)

1. In a separate directory, download the repositories:
    `git clone https://github.com/BayesWatch/cinic-10.git`
    `git clone https://github.com/huyvnphan/PyTorch_CIFAR10.git`
2. `cd` to root and `mkdir external_code`
3. `cd external_code`
4. Create symlinks from the downloaded repositories to inside external code
    `ln -s /path/to/cinic-10 ./`
    `ln -s /path/to/PyTorch_CIFAR10 ./`

### Extract ImageNet Images from CINIC-10
1. From `external_code/cinic-10`, create a symlink to `data/cinic-10`:
    `ln -s /path/to/data/cinic-10 ./` so the path from the root of the repository to the CINIC-10 dataset is `external_code/cinic-10/data/cinic-10`
2. Open the notebook in `external_code/cinic-10/notebooks/imagenet-extraction.ipynb`
3. Verify the paths are correct and run the notebook.

# TEST CHANGE 1