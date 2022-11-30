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
from utils import prepare_device, copy_file, read_paths
from utils.edit_utils import prepare_edit_data


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def knn(K, data_loader, model, base_image=None, data_type='features'):
    '''
    Obtain nearest neighbors for each image in data loader and base image (if not None)

    Arg(s):
        K : int
            how many neighbors to calculate
        data_loader : torch.utils.DataLoader
            shuffle should be false
        model : torch.nn.module
            model
        base_image : torch.tensor or None
            specific image to calculate neighbors for
        data_type : str
            for what data we want to calculate KNN for -- features, logits, images
    '''
    assert data_type in ['features', 'logits', 'images'], "Unsupported data type {}".format(data_type)
    assert not model.training
    assert model.__class__.__name__ == 'CIFAR10PretrainedModelEdit'

    all_data = []
    return_paths = data_loader.get_return_paths()
    context_model = model.context_model()

    with torch.no_grad():
        # First element in all_data will be the base_image representation if it's not None
        base_image.cuda()
        context_model(base_image.cuda())
        base_data = model.get_features(base_image)
        for idx, item in enumerate(tqdm(data_loader)):
            if return_paths:
                image, _, path = item
            else:
                image, _ = item
            image = image.to(device)
            # If we only want images, don't bother running model
            if data_type == 'images':
                all_data.append(image)
                continue
            elif data_type == 'features':
                features = model.target_model(image)
                all_data.append(features)
                continue
            else:
                logits = model(image)
                all_data.append(logits)

        # TODO: if base image is not none, forward and append it

    # Concatenate and convert to numpy
    all_data = torch.cat(all_data, dim=0)
    all_data = all_data.cpu().numpy()



def main(config):
    logger = config.get_logger('train')
    assert config.config['method'] == 'edit', "Invalid method '{}'. Must be 'edit'".format(config.config['method'])

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
    val_data_loader = config.init_obj('data_loader', module_data, split='valid')
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

    # Set up editor
    editor_args = config.config['editor']['args']
    editor_args['arch'] = config.config['arch']['args']['type']

    editor = Editor(
        model=model,
        val_data_loader=val_data_loader,
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
    val_data_name = val_data_loader.get_data_name()
    model_arch = model.get_type()
    layernum = editor.get_layernum()
    cache_dir = os.path.join('cache', val_data_name, "{}-{}".format(model_arch, layernum))

    # Perform edit
    editor.edit(
        edit_data=edit_data,
        model=model,
        cache_dir=cache_dir)

    # Evaluate again on test set
    post_edit_log = predict(
        data_loader=test_data_loader,
        model=model,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        device=device)
    logger.info("Metrics after editing: {}".format(post_edit_log))

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
    print(type(args))
    config = ConfigParser.from_args(args, options)
    # main(config)
