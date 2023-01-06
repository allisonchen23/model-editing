import argparse
import collections
import torch
import numpy as np
import os, sys

from test import predict
sys.path.insert(0, 'src')
# import data_loader.data_loaders as module_data
import datasets.datasets as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer.editor import Editor
from parse_config import ConfigParser
from utils.model_utils import prepare_device
from utils import read_lists
from utils.edit_utils import prepare_edit_data
from utils.analysis import knn


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')
    assert config.config['method'] == 'edit', "Invalid method '{}'. Must be 'edit'".format(config.config['method'])
    K = config.config['editor']['K']  # for KNN
    save_dir = str(config.save_dir)

    # General arguments for data loaders
    dataset_args = config.config['dataset_args']
    data_loader_args = config.config['data_loader']['args']

    # build model architecture, then print to console
    config.config['arch'].update()
    layernum = config.config['layernum']
    model = config.init_obj('arch', module_arch, layernum=layernum)


    logger.info("Created {} model with {} trainable parameters".format(config.config['arch']['type'], model.get_n_params()))
    if model.get_checkpoint_path() != "":
        logger.info("Restored weights from {}".format(model.get_checkpoint_path()))
    else:
        logger.info("Training from scratch.")

    # Load in lists for test data loaders
    # test_image_paths = read_lists(config.config['dataset_paths']['test_images'])
    # test_labels = read_lists(config.config['dataset_paths']['test_labels'])

    # test_data_loader = torch.utils.data.DataLoader(
    #     module_data.CINIC10Dataset(
    #         data_dir="",
    #         image_paths=test_image_paths,
    #         labels=test_labels,
    #         return_paths=False,
    #         **dataset_args
    #     ),
    #     **data_loader_args
    # )
    # # Create test data loader for metric calculations
    # logger.info("Created test data loader")

    # Provide dataloader to perform KNN
    val_image_paths = read_lists(config.config['dataset_paths']['valid_images'])
    val_labels = read_lists(config.config['dataset_paths']['valid_labels'])
    val_paths_data_loader = torch.utils.data.DataLoader(
        module_data.CINIC10Dataset(
            data_dir="",
            image_paths=val_image_paths,
            labels=val_labels,
            return_paths=True,
            **dataset_args
        ),
        **data_loader_args
    )
    logger.info("Created validation data loader for metric and KNN calculations")

    # Prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()  # model should always be in eval() for editing

    # Get function handles for loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # Run initial accuracy check on unedited model
    pre_edit_log = predict(
        data_loader=val_paths_data_loader,
        model=model,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        device=device)

    # Log pre-edit results and save to torch file
    logger.info("Metrics before editing: {}".format(pre_edit_log))
    metric_save_path = os.path.join(save_dir, "pre_edit_test_metrics.pth")
    torch.save(pre_edit_log, metric_save_path)
    logger.info("Saved pre-edit metrics to {}".format(metric_save_path))

    # Prepare data for edit
    key_paths_file = config.config['editor']['key_paths_file']
    key_image_paths = read_lists(key_paths_file)
    value_paths_file = config.config['editor']['value_paths_file']
    value_image_paths = read_lists(value_paths_file)
    mask_paths_file = config.config['editor']['mask_paths_file']


    if mask_paths_file != "":
        mask_paths = read_lists(mask_paths_file)
    else:
        mask_paths = None

    logger.info("Key images: {}".format(key_image_paths))
    logger.info("Value images: {}".format(value_image_paths))
    logger.info("Masks: {}".format(mask_paths))

    edit_data = prepare_edit_data(
        key_image_paths=key_image_paths,
        value_image_paths=value_image_paths,
        mask_paths=mask_paths,
        image_size=(32, 32))
    logger.info("Prepared data for editing")

    if K > 0:
        # # Provide dataloader to perform KNN
        # val_image_paths = read_lists(config.config['dataset_paths']['valid_images'])
        # val_labels = read_lists(config.config['dataset_paths']['valid_labels'])
        # val_paths_data_loader = torch.utils.data.DataLoader(
        #     module_data.CINIC10Dataset(
        #         data_dir="",
        #         image_paths=val_image_paths,
        #         labels=val_labels,
        #         return_paths=True,
        #         **dataset_args
        #     ),
        #     **data_loader_args
        # )
        # logger.info("Created validation data loader for KNN calculations")
        # Concatenate key and value images together
        # First is keys, second is values
        # labels of 'modified_imgs' and 'imgs' are misleading but from the original Editing a Classifier repo
        anchor_images = torch.cat([edit_data['modified_imgs'], edit_data['imgs']], dim=0)
        pre_edit_knn_save_path = os.path.join(save_dir, "pre_edit_{}-nn.pth".format(K))
        logger.info("Performing KNN on validation dataset")
        pre_edit_knn = knn(
            K=K,
            data_loader=val_paths_data_loader,
            model=model,
            anchor_image=anchor_images,
            data_types=['features', 'logits', 'images'],
            device=device,
            save_path=pre_edit_knn_save_path)
        logger.info("Saving pre-edit KNN results with K={} to {}".format(K, pre_edit_knn_save_path))


    # Always use the dummy val_data_loader for covariance calculation
    covariance_image_paths = read_lists(config.config['covariance_dataset']['images'])
    covariance_labels = read_lists(config.config['covariance_dataset']['labels'])

    covariance_data_loader = torch.utils.data.DataLoader(
        module_data.CINIC10Dataset(
            data_dir="",
            image_paths=covariance_image_paths,
            labels=covariance_labels,
            **dataset_args
        ),
        **data_loader_args
    )
    logger.info("Created dataloader for covariance matrix from {}".format(config.config['covariance_dataset']['images']))

    # Set up editor
    editor_args = config.config['editor']['args']
    editor_args['arch'] = config.config['arch']['args']['type']

    editor = Editor(
        val_data_loader=covariance_data_loader,
        **editor_args)

    # Create path for caching directory based on
    #   (1) validation data dir
    #   (2) context model -- architecture, layer number
    val_data_name = config.config['covariance_dataset']['name']
    model_arch = model.get_type()
    # layernum = editor.get_layernum()
    cache_dir = os.path.join('cache', val_data_name, "{}-{}".format(model_arch, layernum))
    logger.info("Looking for covariance matrix weights in {}".format(cache_dir))
    # Perform edit
    editor.edit(
        edit_data=edit_data,
        model=model,
        cache_dir=cache_dir)

    model.save_model(save_path=os.path.join(config._save_dir, "edited_model.pth"))
    # Evaluate again on test set
    logger.info("Evaluating edited model on test set...")
    post_edit_log = predict(
        data_loader=val_paths_data_loader,
        model=model,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        device=device)

    # Log post-edit results and save to torch file
    logger.info("Metrics after editing: {}".format(post_edit_log))
    metric_save_path = os.path.join(save_dir, "post_edit_test_metrics.pth")
    torch.save(post_edit_log, metric_save_path)
    logger.info("Saved post-edit metrics to {}".format(metric_save_path))


    # Perform post edit KNN analysis
    if K > 0:
        # Concatenate key and value images together
        post_edit_knn_save_path = os.path.join(save_dir, "post_edit_{}-nn.pth".format(K))
        logger.info("Performing KNN on validation dataset")
        post_edit_knn = knn(
            K=K,
            data_loader=val_paths_data_loader,
            model=model,
            anchor_image=anchor_images,
            data_types=['features', 'logits', 'images'],
            device=device,
            save_path=post_edit_knn_save_path)
        logger.info("Saving post-edit KNN results with K={} to {}".format(K, post_edit_knn_save_path))

    logger.info("All metrics and KNN results can be found in {}".format(save_dir))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--name'], type=str, target='name')
    ]
    parsed_args = args.parse_args()

    config = ConfigParser.from_args(args, options)
    main(config)
