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
from trainer.editor import get_editor
from parse_config import ConfigParser
from utils.model_utils import prepare_device
from utils import read_lists
from utils.edit_utils import prepare_edit_data
from utils.knn_utils import knn, analyze_knn


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config,
         val_paths_data_loader=None,
         covariance_data_loader=None,
         do_analyze_knn=False):
    # Obtain variables from config
    logger = config.get_logger('train')
    assert config.config['method'] == 'edit', "Invalid method '{}'. Must be 'edit'".format(config.config['method'])
    K = config.config['editor']['K']  # for KNN
    noise_edit = config.config['editor']['args']['noise_edit']
    if noise_edit:
        assert K == 0, "Cannot perform KNN analysis with random edits."

    # Determine edit method based on model type
    if 'Santurkar' in config.config['arch']['type']:
        edit_method = 'eac'
    elif 'Sinitson' in config.config['arch']['type']:
        edit_method = 'enn'

    # Store variables for if we want to perform knn analysis here
    if 'perform_analysis' in config.config['editor']:
        do_analyze_knn = config.config['editor']['perform_analysis']
    else:
        do_analyze_knn = False
    if do_analyze_knn:
        try:
            class_list_path = config.config['class_list_path']
        except:
            raise ValueError("class_list_path not in config file. Aborting")
        try:
            class_list = read_lists(class_list_path)
        except:
            raise ValueError("Unable to read file at {}. Aborting".format(class_list_path))

    save_dir = str(config.save_dir)

    # Prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])

    # build model architecture, then print to console
    config.config['arch'].update()
    layernum = config.config['layernum']
    model = config.init_obj('arch', module_arch,
        # layernum=layernum,
        device=device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()  # model should always be in eval() for editing

    logger.info("Created {} model with {} trainable parameters".format(config.config['arch']['type'], model.get_n_params()))
    model_parameters = model.get_parameters()
    optimizer = config.init_obj('optimizer', torch.optim, model_parameters)

    # Get function handles for loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # Make editable
    if edit_method == 'enn':
        model.make_editable(
            loss_fn=loss_fn)

    if model.get_checkpoint_path() != "":
        logger.info("Restored weights from {}".format(model.get_checkpoint_path()))
    else:
        logger.info("Training from scratch.")

    # General arguments for data loaders
    dataset_args = config.config['dataset_args']
    data_loader_args = config.config['data_loader']['args']

    # Provide dataloader to perform KNN and metric calculation
    if val_paths_data_loader is None:
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
    else:
        logger.info("Using passed in data loader for validation.")


    # Prepare data for edit
    edit_data_args = config.config['editor']['edit_data_args']
    dataset_type = config.config['dataset']['type']
    # log the editor initialization arguments
    logger.info("Edit initialization arguments:")
    for key, val in edit_data_args.items():
        logger.info("{}: {}".format(key, val))
    logger.info("---***---")

    if not noise_edit:
        edit_data = prepare_edit_data(
            edit_method=edit_method,
            **edit_data_args)
        '''
        If edit_method == 'enn': prepare_edit_data() returns a torch.utils.data.DataLoader
        If edit_method == 'eac': prepare_edit_data() returns a dictionary as follows:
            {
                'imgs': torch.tensor(B x C x H x W) [values]
                'modified_imgs': torch.tensor(B x C x H x W) [keys]
                'masks': torch.tensor(B x 1 x H x W) [masks]
            }
        '''

    pre_metric_save_path = os.path.join(save_dir, "pre_edit_metrics.pth")
    pre_logits_save_path = os.path.join(save_dir, "pre_edit_logits.pth")
    if K > 0:
        # Obtain anchor images
        if edit_method == 'eac':
            # for EAC, it is the key and value images
            anchor_images = torch.cat([edit_data['modified_imgs'], edit_data['imgs']], dim=0)
        elif edit_method == 'enn':
            # for ENN, it is all the edit images
            anchor_images = []
            for item in edit_data:
                images, _, _ = item
                anchor_images.append(images)
            anchor_images = torch.cat(anchor_images, dim=0)

        logger.info("Performing pre-edit metric & KNN calculations on validation set.")
        pre_edit_log = knn(
            K=K,
            data_loader=val_paths_data_loader,
            model=model,
            anchor_image=anchor_images,
            data_types=['logits', 'images'],  # ['features', 'logits', 'images']
            metric_fns=metric_fns,
            device=device,
            save_path=None)

        # Save KNN results
        pre_knn_save_path = os.path.join(save_dir, "pre_edit_{}-nn.pth".format(K))
        torch.save(pre_edit_log['knn'], pre_knn_save_path)
        logger.info("Saved pre-edit KNN results with K={} to {}".format(K, pre_knn_save_path))
    else:  # if not performing KNN
        logger.info("Performing pre-edit metric calculations on validation set.")
        pre_edit_log = predict(
            data_loader=val_paths_data_loader,
            model=model,
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            device=device)
    # Save pre edit metrics
    logger.info("Pre-edit metrics: {}".format(pre_edit_log['metrics']))
    torch.save(pre_edit_log['metrics'], pre_metric_save_path)
    logger.info("Saved pre-edit metrics {}".format(pre_metric_save_path))
    # Save logits
    torch.save(pre_edit_log['logits'], pre_logits_save_path)
    logger.info("Saved pre-edit logits to {}".format(pre_logits_save_path))

    # Set up editor
    editor_args = config.config['editor']['args']
    if edit_method == 'eac':
        editor_args['arch'] = config.config['arch']['args']['type']
    # elif edit_method == 'enn':
    #     editor_args.update({
    #         'model': model,
    #         'optimizer': optimizer,
    #         'loss_fn': loss_fn,
    #         })
    #     if 'error' in editor_args:
    #         editor_args.pop('error')
    #         editor_args['error_fn'] = None

    editor = get_editor(
        edit_method=edit_method,
        **editor_args)

    logger.info("Created {} editor.".format(type(editor)))

    # Prepare covariance dataset if necessary
    if not noise_edit:
        if edit_method == 'eac':
            if covariance_data_loader is None:
                if 'covariance_dataset' in config.config and 'images' in config.config['covariance_dataset']:
                    # Always use the dummy val_data_loader for covariance calculation
                    covariance_image_paths = read_lists(config.config['covariance_dataset']['images'])
                    covariance_labels = read_lists(config.config['covariance_dataset']['labels'])

                    covariance_data_loader = torch.utils.data.DataLoader(
                        module_data.CINIC10Dataset(
                            data_dir="",
                            image_paths=covariance_image_paths,
                            labels=covariance_labels,
                            return_paths=False,
                            **dataset_args
                        ),
                        **data_loader_args
                    )
                    val_data_name = config.config['covariance_dataset']['name']

                    logger.info("Created dataloader for covariance matrix from {}".format(config.config['covariance_dataset']['images']))
                else:  # Use identity matrix
                    covariance_data_loader = None
                    val_data_name = "identity"
                    logger.info("No data loader for covariance matrix. Will use identity matrix")
            else:
                val_data_name = config.config['covariance_dataset']['name']
                logger.info("Using passed in covariance data loader.")
    else:
        logger.info("Performing random edit. No covariance matrix needed.")

    # Perform edit
    if not noise_edit:
        if edit_method == 'eac':
            # Create path for caching directory based on
            #   (1) validation data dir
            #   (2) context model -- architecture, layer number
            model_arch = model.get_type()
            cache_dir = os.path.join('cache', val_data_name, "{}-{}".format(model_arch, layernum))
            logger.info("Looking for covariance matrix weights in {}".format(cache_dir))
            editor.edit(
                edit_data=edit_data,
                model=model,
                val_data_loader=covariance_data_loader,
                cache_dir=cache_dir)
        elif edit_method == 'enn':
            edit_results_save_path = os.path.join(save_dir, "edit_results.pth")
            edit_images, edit_labels = edit_data
            edited_model, edit_results = editor.edit(
                model=model,
                inputs=edit_images,
                targets=edit_labels,
                save_path=edit_results_save_path
            )
            logger.info("Edit results from ENN: {}".format(edit_results))
            model.model = edited_model

    else:
        editor.noise_edit(
            model=model
        )

    if not do_analyze_knn and not noise_edit:
        model.save_model(save_path=os.path.join(save_dir, "edited_model.pth"))

    # Perform post edit KNN analysis
    post_metric_save_path = os.path.join(save_dir, "post_edit_metrics.pth")
    post_logits_save_path = os.path.join(save_dir, "post_edit_logits.pth")
    if K > 0:
        # Concatenate key and value images together

        logger.info("Performing post-edit metric & KNN calculations on validation set.")

        post_edit_log = knn(
            K=K,
            data_loader=val_paths_data_loader,
            model=model,
            anchor_image=anchor_images,
            data_types=['features', 'logits', 'images'],
            metric_fns=metric_fns,
            device=device,
            save_path=None)

        logger.info("Post-edit metrics: {}".format(post_edit_log['metrics']))
        # Save metrics
        # post_metric_save_path = os.path.join(save_dir, "post_edit_metrics.pth")
        # torch.save(post_edit_log['metrics'], post_metric_save_path)
        # logger.info("Saved post-edit metrics to {}".format(post_metric_save_path))
        # Save KNN results
        post_knn_save_path = os.path.join(save_dir, "post_edit_{}-nn.pth".format(K))
        torch.save(post_edit_log['knn'], post_knn_save_path)
        logger.info("Saving post-edit KNN results with K={} to {}".format(K, post_knn_save_path))
        # Save logits
        # torch.save(post_edit_log['logits'], post_logits_save_path)
        # logger.info("Saved post-edit logits to {}".format(post_logits_save_path))
    else:  # if not performing KNN
        logger.info("Performing post-edit metric calculations on validation set.")

        post_edit_log = predict(
            data_loader=val_paths_data_loader,
            model=model,
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            device=device)

    logger.info("Post-edit metrics: {}".format(post_edit_log['metrics']))
    # Save post edit metrics
    torch.save(post_edit_log['metrics'], post_metric_save_path)
    logger.info("Saved post-edit metrics {}".format(post_metric_save_path))
    # Save post edit logits
    torch.save(post_edit_log['logits'], post_logits_save_path)
    logger.info("Saved post-edit logits to {}".format(post_logits_save_path))

    # Analyze KNN results here and now
    if do_analyze_knn and K > 0:
        logger.info("Performing KNN analysis...")
        target_class_idx = np.argmax(post_edit_log['knn']['logits']['anchor_data'][0])
        analyze_knn(
            save_dir=save_dir,
            config=config,
            pre_edit_knn=pre_edit_log['knn'],
            post_edit_knn=post_edit_log['knn'],
            edited_model=model,
            knn_analysis_filename='knn_analysis_results.pth',
            target_class_idx=target_class_idx,
            class_list=class_list,
            progress_report_path=None,
            save_plots=True)

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
    main(config, do_analyze_knn=True)
