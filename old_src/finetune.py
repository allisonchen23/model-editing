import torch
import os
import argparse
from pytorch_lightning import Trainer, seed_everything
# Local files
import datasets
import models
from helpers import log

parser = argparse.ArgumentParser()


def finetune_model(  # Dataset related parameters
                   target_dataset_path,
                   normalize,
                   mean,
                   std,
                   batch_size,
                   n_threads,
                   # Model related parameters
                   model_restore_path,
                   model_type,
                   model_source,
                   learning_rate,
                   weight_decay,
                   n_epochs,
                   # Misc
                   device,
                   save_path,
                   is_deterministic):

    log_path = os.path.join(save_path, "log.txt")
    # Log inputs
    log("Is deterministic: {}".format(is_deterministic), log_path)

    if is_deterministic:
        seed_everything(0)
    # Load model checkpoint
    model = models.load_model(
        model_restore_path=model_restore_path,
        model_type=model_type,
        model_source=model_source,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        n_epochs=n_epochs)

    log("Model {} restored".format(model_type), log_path)

    # Load train dataloader
    dataset_paths = {
        'train': os.path.join(target_dataset_path, 'train'),
        'val': os.path.join(target_dataset_path, 'valid'),
        'test': os.path.join(target_dataset_path, 'test')
    }
    dataset = datasets.CINIC10ImageNetDataset(
        dataset_paths=dataset_paths,
        normalize=normalize,
        mean=mean,
        std=std,
        batch_size=batch_size,
        n_threads=n_threads)
    # dataset.setup()
    dataset.setup(stage="all")

    train_dataloader = dataset.train_dataloader()
    print(train_dataloader)
    val_dataloader = dataset.val_dataloader()
    print(val_dataloader)
    # train_dataloader = dataset.dataloader(
    #     type="train",
    #     batch_size=batch_size,
    #     n_threads=n_threads)
    # val_dataloader = dataset.dataloader(
    #     type="val",
    #     batch_size=batch_size,
    #     n_threads=n_threads)

    log("Loaded train and validation dataloaders with following settings:", log_path)
    log("Normalize: {} \tMean: {}\t Standard Deviation: {}".format(
        normalize, mean, std), log_path)
    log("Batch size: {}\tNumber of threads: {}".format(
            batch_size, n_threads), log_path)
    # train_dataloader = torch.utils.data.DataLoader(
    #     datasets.get_dataset(
    #         dataset_path=target_dataset_path,
    #         split="train",
    #         normalize=normalize,
    #         mean=mean,
    #         std=std),
    #     batch_size=batch_size,
    #     num_workers=n_threads,
    #     shuffle=True,
    #     drop_last=False)

    # val_dataloader = torch.utils.data.DataLoader(
    #     datasets.get_dataset(
    #         dataset_path=target_dataset_path,
    #         split="valid",
    #         normalize=normalize,
    #         mean=mean,
    #         std=std),
    #     batch_size=batch_size,
    #     num_workers=n_threads,
    #     shuffle=False,
    #     drop_last=False)

    # Initialize trainer
    trainer = Trainer(
        accelerator=device,
        auto_select_gpus=True,
        gpus=[0],
        max_epochs=n_epochs,
        log_every_n_steps=1000,
        # enable_progress_bar=True,
        deterministic=is_deterministic)

    trainer.fit(
        model=model,
        datamodule=dataset)

    model_save_path = os.path.join(save_path, "{}_finetuned.pt".format(model_type))
    models.save_model(
        model_save_path=model_save_path,
        model=model)

    # trainer.test(
    #     model=model,
    #     dataloaders=val_dataloader,
    #     verbose=True)
    trainer.test()

if __name__ == "__main__":
    # Dataset related arguments
    parser.add_argument("--target_dataset_path", type=str, required=True,
        help="Path to dataset to fine tune on. Must have 'train/valid/test' splits.")
    parser.add_argument("--normalize", action='store_true',
        help="If set, normalize images by mean and standard deviation")
    parser.add_argument("--mean", nargs="+", type=float,
        help="Space delimited list of mean values for RGB channels")
    parser.add_argument("--std", nargs="+", type=float,
        help="Space delimited list of standard deviation values for RGB channels")
    parser.add_argument("--batch_size", type=int, default=128,
        help="Batch size")
    parser.add_argument("--n_threads", type=int, default=8,
        help="Number of threads")

    # Model related arguments
    parser.add_argument("--model_restore_path", type=str, required=True,
        help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, required=True,
        help="Model architecture as string")
    parser.add_argument("--model_source", type=str, required=True,
        help="Where model is coming from. Choose from ['pretrained']")
    parser.add_argument("--learning_rate", type=float, default=1e-2,
        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
        help="Value for weight decay")
    parser.add_argument("--n_epochs", type=int, default=15,
        help="Number of epochs to fine tune for")

    # Misc
    parser.add_argument("--device", type=str, default='gpu',
        help="Type of device to use ('gpu' or 'cpu')")
    parser.add_argument("--save_path", type=str, required=True,
        help="Directory path to log results and save checkpoints")
    parser.add_argument("--is_deterministic", action='store_true',
        help="If set, normalize images by mean and standard deviation")

    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    finetune_model(
        # Dataset parameters
        target_dataset_path=args.target_dataset_path,
        batch_size=args.batch_size,
        normalize=args.normalize,
        mean=args.mean,
        std=args.std,
        n_threads=args.n_threads,
        # Model parameters
        model_restore_path=args.model_restore_path,
        model_type=args.model_type,
        model_source=args.model_source,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        # Misc
        save_path=args.save_path,
        device=args.device,
        is_deterministic=args.is_deterministic)

