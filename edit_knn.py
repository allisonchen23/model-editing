import argparse
import collections
import torch
import numpy as np
import os, sys

from test import predict
sys.path.insert(0, 'src')
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer.editor import Editor
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device, copy_file, read_lists
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

    # build model architecture, then print to console
    config.config['arch'].update()
    layernum = config.config['layernum']
    model = config.init_obj('arch', module_arch, layernum=layernum)


    logger.info("Created {} model with {} trainable parameters".format(config.config['arch']['type'], model.get_n_params()))
    if model.get_checkpoint_path() != "":
        logger.info("Restored weights from {}".format(model.get_checkpoint_path()))
    else:
        logger.info("Training from scratch.")

    # Create validation and test dataloaders
    val_paths_data_loader = config.init_obj(
        'data_loader', 
        module_data, 
        split='valid', 
        return_paths=True)
    val_data_loader = config.init_obj(
        'data_loader', 
        module_data, 
        split='valid')
    test_data_loader = config.init_obj('data_loader', module_data, split='test')

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
        data_loader=test_data_loader,
        model=model,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        device=device)
    logger.info("Metrics before editing: {}".format(pre_edit_log))

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
    
    if K > 0:
        # Concatenate key and value images together
        # First is keys, second is values
        anchor_images = torch.cat([edit_data['modified_imgs'], edit_data['imgs']], dim=0)
        pre_edit_knn_save_path = os.path.join(config._save_dir, "pre_edit_{}-nn.pth".format(K))
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

    # Set up editor
    editor_args = config.config['editor']['args']
    editor_args['arch'] = config.config['arch']['args']['type']

    editor = Editor(
        # model=model,
        val_data_loader=val_data_loader,
        **editor_args)


    # Create path for caching directory based on
    #   (1) validation data dir
    #   (2) context model -- architecture, layer number
    val_data_name = val_data_loader.get_data_name()
    model_arch = model.get_type()
    # layernum = editor.get_layernum()
    cache_dir = os.path.join('cache', val_data_name, "{}-{}".format(model_arch, layernum))

    # Perform edit
    editor.edit(
        edit_data=edit_data,
        model=model,
        cache_dir=cache_dir)

    model.save_model(save_path=os.path.join(config._save_dir, "edited_model.pth"))
    # Evaluate again on test set
    logger.info("Evaluating edited model on test set...")
    post_edit_log = predict(
        data_loader=test_data_loader,
        model=model,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        device=device)
    logger.info("Metrics after editing: {}".format(post_edit_log))
    
    # Perform post edit KNN analysis
    if K > 0:
        # # Concatenate key and value images together
        # anchor_images = torch.cat([edit_data['modified_imgs'], edit_data['imgs']], dim=0)
        post_edit_knn_save_path = os.path.join(config._save_dir, "post_edit_{}-nn.pth".format(K))
        logger.info("Performing KNN on validation dataset")
        pre_edit_knn = knn(
            K=K,
            data_loader=val_paths_data_loader,
            model=model,
            anchor_image=anchor_images, 
            data_types=['features', 'logits', 'images'],
            device=device,
            save_path=post_edit_knn_save_path)
        logger.info("Saving post-edit KNN results with K={} to {}".format(K, post_edit_knn_save_path))


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
