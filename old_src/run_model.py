import os, sys
import torch
import argparse
from pytorch_lightning import Trainer
sys.path.insert(0, os.path.join('external_code', 'PyTorch_CIFAR10', 'cifar10_models'))
import datasets
import helpers

parser = argparse.ArgumentParser()

def run_model(dataloader,
              trainer,
              model_restore_path,
              model_type,
              return_predictions=False,
              verbose=False):
    '''
    Run the model on the dataloader specified.
    If return_predictions is true, return tensor of all predictions.
    Else, return the accuracy
    '''

    model = helpers.CIFAR10Module(
        model_type=model_type)
    checkpoint = torch.load(model_restore_path)
    model.model.load_state_dict(checkpoint)
    if verbose:
        print("Loaded model {} from {}".format(model_type, model_restore_path))

    if return_predictions:
        return trainer.predict(
            model=model,
            dataloaders=dataloader)
    else:
        return trainer.test(
            model=model,
            dataloaders=dataloader,
            verbose=True)


if __name__ == "__main__":
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--model_restore_path", type=str, required=True, help="Path to stored checkpoint")
    parser.add_argument("--model_type", type=str, required=True, help="String of what model type checkpoint is")
    parser.add_argument("--data_split", type=str, default='test', help="Must be in ['train', 'test', 'valid']")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--normalize", action='store_true', help="If set, normalize images by mean and standard deviation")
    parser.add_argument("--mean", nargs="+", type=float, help="Space delimited list of mean values for RGB channels")
    parser.add_argument("--std", nargs="+", type=float, help="Space delimited list of standard deviation values for RGB channels")
    parser.add_argument("--n_threads", type=int, default=8, help="Number of threads")
    parser.add_argument("--device", type=str, default='cuda', help="Type of device to use ('cuda' or 'cpu')")
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

    run_model(
        dataset_path=args.dataset_path,
        model_restore_path=args.model_restore_path,
        model_type=args.model_type,
        data_split=args.data_split,
        batch_size=args.batch_size,
        normalize=args.normalize,
        mean=args.mean,
        std=args.std,
        n_threads=args.n_threads,
        device=args.device,
        gpu_ids=args.gpu_ids,
        verbose=args.verbose)
