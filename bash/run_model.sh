# export CUDA_VISIBLE_DEVICES=0,1

python src/run_model.py \
--dataset_path data/cinic-10-imagenet \
--model_restore_path external_code/PyTorch_CIFAR10/cifar10_models/state_dicts/vgg16_bn.pt \
--model_type vgg16_bn \
--data_split test \
--batch_size 128 \
--normalize \
--mean 0.4914 0.4822 0.4465 \
--std 0.2471 0.2435 0.2616 \
--n_threads 8 \
--device gpu \
--gpu_ids 0 1 \
--verbose