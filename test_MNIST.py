import argparse
import torch
from tqdm import tqdm
import sys
import os
from mlxtend.evaluate import accuracy_score
sys.path.insert(0, 'src')

import datasets.datasets as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from utils import read_lists, ensure_dir
from parse_config import ConfigParser

def predict(data_loader,
            model,
            loss_fn,
            metric_fns,
            device,
            output_save_path=None,
            log_save_path=None):
    '''
    Run the model on the data_loader, calculate metrics, and log

    Arg(s):
        data_loader : torch Dataloader
            data to test on
        model : torch.nn.Module
            model to run
        loss_fn : module
            loss function
        metric_fns : list[model.metric modules]
            list of metric functions
        device : torch.device
            GPU device
        output_save_path : str or None
            if not None, save model_outputs to save_path
        log_save_path : str or None
            if not None, save metrics to save_path


    Returns :
        log : dict{} of metrics
    '''

    # Hold data for calculating metrics
    outputs = []
    targets = []

    # Ensure model is in eval mode
    if model.training:
        model.eval()

    with torch.no_grad():
        for idx, item in enumerate(tqdm(data_loader)):
            if len(item) == 3:
                data, target, path = item
            else:
                data, target = item
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Store outputs and targets
            outputs.append(output)
            targets.append(target)

    # Concatenate predictions and targets
    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    # Calculate loss
    loss = loss_fn(outputs, targets).item()
    n_samples = len(data_loader.sampler)
    log = {'loss': loss}

    # Calculate predictions based on argmax
    predictions = torch.argmax(outputs, dim=1)

    # Move predictions and target to cpu and convert to numpy to calculate metrics
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    # Calculate metrics
    log = module_metric.compute_metrics(
        metric_fns=metric_fns,
        prediction=predictions,
        target=targets)

    if output_save_path is not None:
        ensure_dir(os.path.dirname(output_save_path))
        torch.save(outputs, output_save_path)

    if log_save_path is not None:
        ensure_dir(os.path.dirname(log_save_path))
        torch.save(log, log_save_path)

    return_data = {
        'metrics': log,
        'logits': outputs
    }
    return return_data

def predict_with_bump(data_loader,
                      model,
                      target_class_idx,
                      bump_amount,
                      loss_fn,
                      metric_fns,
                      device,
                      output_save_path=None,
                      log_save_path=None):
    '''
    Run the model on the data_loader, calculate metrics, and log

    Arg(s):
        data_loader : torch Dataloader
            data to test on
        model : torch.nn.Module
            model to run
        loss_fn : module
            loss function
        metric_fns : list[model.metric modules]
            list of metric functions
        device : torch.device
        output_save_path : str or None
            if not None, save model_outputs to save_path
        log_save_path : str or None
            if not None, save metrics to save_path

    Returns :
        log : dict{} of metrics
    '''

    # Hold data for calculating metrics
    outputs = []
    targets = []

    # Ensure model is in eval mode
    if model.training:
        model.eval()

    with torch.no_grad():
        for idx, item in enumerate(tqdm(data_loader)):
            if len(item) == 3:
                data, target, path = item
            else:
                data, target = item
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Store outputs and targets
            outputs.append(output)
            targets.append(target)

    # Concatenate predictions and targets
    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    # Adjust output softmax by bump amount
    outputs[:, target_class_idx] += bump_amount

    # Calculate loss
    loss = loss_fn(outputs, targets).item()
    n_samples = len(data_loader.sampler)


    # Calculate predictions based on argmax
    predictions = torch.argmax(outputs, dim=1)

    # Move predictions and target to cpu and convert to numpy to calculate metrics
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    # Calculate metrics
    log = module_metric.compute_metrics(
        metric_fns=metric_fns,
        prediction=predictions,
        target=targets)

    # Add bump amount to log
    log.update({'loss': loss})
    log.update({'bump_amount': bump_amount})

    if output_save_path is not None:
        ensure_dir(os.path.dirname(output_save_path))
        torch.save(outputs, output_save_path)

    if log_save_path is not None:
        ensure_dir(os.path.dirname(log_save_path))
        torch.save(log, log_save_path)

    return {
        'metrics': log,
        'logits': outputs
    }


def main(config, test_data_loader=None):
    logger = config.get_logger('test')
    logger.info("Results saved to {}".format(os.path.dirname(config.log_dir)))

    # General arguments for data loaders
    dataset_args = config.config['dataset']['args']
    data_loader_args = config.config['data_loader']['args']
    # The architecture of the Edited model already normalizes
    if config.config['arch']['type'] == "ModelWrapperSanturkar":
        dataset_args['normalize'] = False
        logger.warning("Using edited model architecture. Overriding normalization for dataset to False.")

    # setup data_loader instances
    if test_data_loader is None:
        test_dataset = config.init_obj('dataset', module_data)
        test_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            **data_loader_args
        )
        logger.info("Created test ({} images) dataloader from {}.".format(
            len(test_dataset),
            test_dataset.dataset_dir))

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info("Created {} model with {} trainable parameters".format(config.config['arch']['type'], model.get_n_params()))

    # First priority is check for resumed path
    if config.resume is not None:
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        logger.info("Restored weights from {}".format(config.resume))
    elif model.get_checkpoint_path() != "":
        logger.info("Restored weights from {}".format(model.get_checkpoint_path()))
    else:
        raise ValueError("No checkpoint provided to restore model from")

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = []
    for met in config['metrics']:
        metric_fns.append(getattr(module_metric, met))
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Save results as a pickle file for easy deserialization
    metric_save_path = os.path.join(str(config.log_dir), 'test_metrics.pth')
    logits_save_path = os.path.join(str(config.log_dir), 'logits.pth')
    log = predict(
        data_loader=test_data_loader,
        model=model,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        device=device,
        log_save_path=metric_save_path,
        output_save_path=logits_save_path)
    for log_key, log_item in log.items():
        logger.info("{}: {}".format(log_key, log_item))
    # logger.info(log)
    logger.info("Saved test metrics to {}".format(metric_save_path))

    # Final message
    logger.info("Access results at {}".format(os.path.dirname(config.log_dir)))
    return log


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
