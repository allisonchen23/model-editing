import argparse
import collections
import torch
import numpy as np
import os, sys
sys.path.insert(0, 'src')
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer.editor import Editor
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device, copy_file, read_paths
from utils.edit_utils import prepare_edit_data


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')
    assert config.config['method'] == 'edit', "Invalid method '{}'. Must be 'edit'".format(config.config['method'])

    # build model architecture, then print to console
    config.config['arch'].update()
    model = config.init_obj('arch', module_arch)

    logger.info("Created {} model with {} trainable parameters".format(config.config['arch']['type'], model.get_n_params()))
    if model.get_checkpoint_path() != "":
        logger.info("Restored weights from {}".format(model.get_checkpoint_path()))
    else:
        logger.info("Training from scratch.")

    # Create validation dataloader
    val_dataloader = config.init_obj('data_loader', module_data, split='valid')


    # Set up editor
    editor_args = config.config['editor']['args']
    editor_args['arch'] = config.config['arch']['args']['type']

    editor = Editor(
        model=model,
        val_dataloader=val_dataloader,
        **editor_args)

    # Prepare data for edit
    key_paths_file = config.config['editor']['key_paths_file']
    key_image_paths = read_paths(key_paths_file)
    value_paths_file = config.config['editor']['value_paths_file']
    value_image_paths = read_paths(value_paths_file)
    mask_paths_file = config.config['editor']['mask_paths_file']
    if mask_paths_file != "":
        mask_paths = read_paths(mask_paths_file)
    else:
        mask_paths = None

    edit_data = prepare_edit_data(
        key_image_paths=key_image_paths,
        value_image_paths=value_image_paths,
        mask_paths=mask_paths)

    # Create path for caching directory based on
    #   (1) validation data dir
    #   (2) context model -- architecture, layer number
    val_data_name = val_dataloader.get_data_name()
    model_arch = model.get_type()
    layernum = editor.get_layernum()
    cache_dir = os.path.join('cache', val_data_name, "{}-{}".format(model_arch, layernum))
    # print(editor_args)

    # Perform edit
    editor.edit(
        edit_data=edit_data,
        cache_dir=cache_dir)

'''
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    if config.config['lr_scheduler']['type'] != "None":
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    else:
        lr_scheduler = None

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=train_data_loader,
                      valid_data_loader=val_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()
'''

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
