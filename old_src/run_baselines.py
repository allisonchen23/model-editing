import os, sys
import torch
import argparse
from pytorch_lightning import Trainer, seed_everything
sys.path.insert(0, os.path.join('external_code', 'PyTorch_CIFAR10', 'cifar10_models'))
import datasets
from run_model import run_model
from helpers import log, CIFAR10Module

MODEL_NAMES = ['vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
               'resnet18', 'resnet34', 'resnet50',
               'densenet121', 'densenet161', 'densenet169',
               'mobilenet_v2', 'googlenet', 'inception_v3']
EXPECTED_CIFAR10_ACC = {
    "vgg11_bn": 92.39,
    "vgg13_bn": 94.22,
    "vgg16_bn": 94.00,
    "vgg19_bn": 93.95,
    "resnet18": 93.07,
    "resnet34": 93.34,
    "resnet50": 93.65,
    "densenet121": 94.06,
    "densenet161": 94.07,
    "densenet169": 94.05,
    "mobilenet_v2": 93.91,
    "googlenet": 92.85,
    "inception_v3": 93.74,
}
parser = argparse.ArgumentParser()
# To make results reproducible
seed_everything(0, workers=True)

def run_baselines(restore_model_dir,
                  dataset_path,
                  log_path,
                  data_split='test',
                  batch_size=128,
                  normalize=False,
                  mean=None,
                  std=None,
                  n_threads=8,
                  device='gpu',
                  verbose=False,):
    dataloader = torch.utils.data.DataLoader(
        datasets.get_dataset(
            dataset_path=dataset_path,
            split=data_split,
            normalize=normalize,
            mean=mean,
            std=std),
        batch_size=batch_size,
        num_workers=n_threads,
        shuffle=False,
        drop_last=False)
    if verbose:
        print("Created {} dataloader.".format(data_split))

    # Initialize trainer
    trainer = Trainer(
        accelerator=device,
        auto_select_gpus=True,
        gpus=[0],
        log_every_n_steps=1000,
        enable_progress_bar=True,
        deterministic=True)
    if verbose:
        print("Initialized Trainer")

    for model_name in MODEL_NAMES:
        model_path = os.path.join(restore_model_dir, model_name + ".pt")
        results = run_model(
            dataloader=dataloader,
            trainer=trainer,
            model_restore_path=model_path,
            model_type=model_name,
            return_predictions=False,
            verbose=True)

        log("Model type: {} Results: {}".format(model_name, results), log_path)

def store_predictions(restore_model_dir,
                      dataset_path,
                      checkpoint_dir,
                      log_path,
                      data_split='test',
                      batch_size=128,
                      normalize=False,
                      mean=None,
                      std=None,
                      n_threads=8,
                      device='gpu',
                      verbose=False,):
    dataloader = torch.utils.data.DataLoader(
        datasets.get_dataset(
            dataset_path=dataset_path,
            split=data_split,
            normalize=normalize,
            mean=mean,
            std=std),
        batch_size=batch_size,
        num_workers=n_threads,
        shuffle=False,
        drop_last=False)
    if verbose:
        print("Created {} dataloader.".format(data_split))

    # Initialize trainer
    trainer = Trainer(
        accelerator=device,
        auto_select_gpus=True,
        gpus=[0],
        log_every_n_steps=1000,
        enable_progress_bar=True,
        deterministic=True)
    if verbose:
        print("Initialized Trainer")

    for model_name in MODEL_NAMES:
        model_path = os.path.join(restore_model_dir, model_name + ".pt")
        results = run_model(
            dataloader=dataloader,
            trainer=trainer,
            model_restore_path=model_path,
            model_type=model_name,
            return_predictions=True,
            verbose=True)

        save_path = os.path.join(checkpoint_dir, model_name + "preds.pt")
        checkpoint = {"preds": results}
        torch.save(checkpoint, save_path)
        log("Model type: {} Saving results to: {}".format(model_name, results), log_path)


if __name__ == "__main__":
    parser.add_argument("--restore_model_dir", type=str, required=True, help="Path to stored checkpoints directory")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to directory for logs")
    parser.add_argument("--data_split", type=str, default='test', help="Must be in ['train', 'test', 'valid']")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--normalize", action='store_true', help="If set, normalize images by mean and standard deviation")
    parser.add_argument("--mean", nargs="+", type=float, help="Space delimited list of mean values for RGB channels")
    parser.add_argument("--std", nargs="+", type=float, help="Space delimited list of standard deviation values for RGB channels")
    parser.add_argument("--n_threads", type=int, default=8, help="Number of threads")
    parser.add_argument("--device", type=str, default='cuda', help="Type of device to use ('gpu' or 'cpu')")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0], help="Space delimited list of GPU IDs. Use -1 to denote no GPUS")
    parser.add_argument("--verbose", action="store_true", help="Verbose printing if true")

    args = parser.parse_args()

    # Checks
    if -1 in args.mean:
        args.mean = None
    if -1 in args.std:
        args.std = None
    if -1 in args.gpu_ids:
        args.gpu_ids = None

    log_path = os.path.join(args.checkpoint_dir, 'results.txt')
    run_baselines(
        restore_model_dir=args.restore_model_dir,
        dataset_path=args.dataset_path,
        log_path=log_path,
        data_split=args.data_split,
        batch_size=args.batch_size,
        normalize=args.normalize,
        mean=args.mean,
        std=args.std,
        n_threads=args.n_threads,
        device=args.device,
        verbose=args.verbose)