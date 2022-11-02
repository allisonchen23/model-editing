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
