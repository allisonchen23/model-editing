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

### Running Experiments and Processing

#### Trials
1. Run the trials from `edit_segmented_analyze_trials.ipynb` or `edit_segmented_analyze_trials_all_classes.ipynb`

#### Bump Edits
1. In the notebook `bump_noise_model_editing.ipynb`, you only need to run the cells under `Get the Target Class Distribution Across All Edits for Specific Class` one time.
2. Run the remaining cells. The first group calculates how much to bump in order to match the values of class predictions. The second group replicates the distribution for this class. From the last cell, you will get a `trial_paths.txt` file to use for the next steps.

#### Processing results and creating HTML

2. For each trial, run `python src/utils/results_to_csv.py --trial_paths_path <path/to/trial_paths.txt>`. This will output a file `results_table.csv` in the same directory as `trial_paths.txt`.
3. To generate summary metric graphs + class distribution graphs, run the notebook `csv_analysis.ipynb` up to and including the cell under 'Print Summaries'
4. To generate summary HTML pages, create a config file similar to `configs/summary_html_template.json` and run `python build_html.py --config <path/to/config>`. Replace any empty strings and "[]" with the appropriate value