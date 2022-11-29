import argparse
import torch
from tqdm import tqdm
import sys
sys.path.insert(0, 'src')

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

def predict(data_loader, model, loss_fn, metric_fns, device):
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
        logger : logger or None

    Returns :
        log : dict{} of metrics
    '''

    total_loss = 0.0
    metrics = module_metric.Metrics(metric_fns)
    # total_metrics = torch.zeros(len(metric_fns))
    # total_metrics = module_metric.initialize_total_metrics(metric_fns)
    return_paths = data_loader.get_return_paths()

    with torch.no_grad():
        for idx, item in enumerate(tqdm(data_loader)):
            if return_paths:
                data, target, path = item
            else:
                data, target = item
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
<<<<<<< HEAD
            metrics.update(output, target)
=======
            total_metrics.update(output, target)
>>>>>>> 48aa8e6 (need to convert stuff between torch and numpy for 'per_class_counts()')
            # for metric_idx, metric in enumerate(metric_fns):
            #     total_metrics[metric_idx] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    total_metrics = metrics.get_total_metrics()
    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    log.update(total_metrics)
    # if logger is not None:
    #     logger.info(log)
    _, idx_to_class_map = data_loader.get_class_idx_maps()
    per_class_accuracies = {}
    for idx, accuracy in enumerate(total_metrics['per_class_acc']):
        per_class_accuracies[idx_to_class_map[idx]] = accuracy.item()
    log.update({'per_class_acc_name': per_class_accuracies})

    return log


def main(config, test_data_loader=None):
    logger = config.get_logger('test')

    # setup data_loader instances
    if test_data_loader is None:
        test_data_loader = config.init_obj('data_loader', module_data, split='test')
        logger.info("Created test data loader from '{}'".format(test_data_loader.get_data_dir()))

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
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    log = predict(
        data_loader=test_data_loader,
        model=model,
        loss_fn=loss_fn,
        metric_fns=metric_fns,
        device=device)
    for log_key, log_item in log.items():
        logger.info("{}: {}".format(log_key, log_item))
    # logger.info(log)

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
