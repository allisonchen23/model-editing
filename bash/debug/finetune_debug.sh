python src/finetune.py \
--target_dataset_path data/cinic-10-imagenet \
--normalize \
--mean 0.4914 0.4822 0.4465 \
--std 0.2471 0.2435 0.2616 \
--n_threads 8 \
--model_restore_path external_code/PyTorch_CIFAR10/cifar10_models/state_dicts/vgg16_bn.pt \
--model_type vgg16_bn \
--model_source pretrained \
--learning_rate 1e-2 \
--weight_decay 0.0 \
--n_epochs 15 \
--save_path results/finetune/pretrained_vgg16bn/debug \
--device gpu