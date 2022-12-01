import argparse
import collections
import torch
import numpy as np
import sys
sys.path.insert(0, 'src')
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device, copy_file


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config, train_data_loader=None, val_data_loader=None):
    logger = config.get_logger('train')

    # setup data_loader instances
    if train_data_loader is None and val_data_loader is not None:
        raise ValueError("No data loader passed for validation")
    elif train_data_loader is not None and val_data_loader is None:
        raise ValueError("No data loader passed for training")
    elif train_data_loader is None and val_data_loader is None:
        train_data_loader = config.init_obj('data_loader', module_data, split='train')
        val_data_loader = config.init_obj('data_loader', module_data, split='valid')

        logger.info("Created train and validation dataloaders for {}".format(config.config['data_loader']['type']))
        logger.info("Train data folder: {}".format(train_data_loader.get_data_dir()))
        logger.info("Validation data folder: {}".format(val_data_loader.get_data_dir()))

        logger.info("Training augmentations: {}".format(
            config.config['data_loader']['args']['augmentations']))

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info("Created {} model with {} trainable parameters".format(config.config['arch']['type'], model.get_n_params()))
    if model.get_checkpoint_path() != "":
        logger.info("Restored weights from {}".format(model.get_checkpoint_path()))
    else:
        logger.info("Training from scratch.")

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

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
                      lr_scheduler=lr_scheduler,
                      valid_epoch=config.config['trainer']['valid_epoch'])

    trainer.train()


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
