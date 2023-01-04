import torch
import numpy as np

def quick_predict(model, image_path, device):
    '''
    Return model output for image(s) at image_path
    Arg(s):
        model : torch.nn.Module
            model to predict
        image_path : str or list[str]
            image(s) to predict for
        device : torch.device
            device that the model is located on
    '''
    # Load image
    if type(image_path) == str:
        image = load_image(image_path)
        # Expand to 1 x C x H x W and convert to tensor
        image = np.expand_dims(image, axis=0)
    elif type(image_path) == list:
        image = []
        for path in image_path:
            image.append(load_image(path))

        image = np.stack(image, axis=0)
    else:
        raise ValueError("Unsupported type {} for image_path".format(type(image_path)))

    # Convert to torch
    image = torch.from_numpy(image)
    # Convert from double -> float and switch to device
    image = image.type(torch.FloatTensor).to(device)

    # Pass through model
    with torch.no_grad():
        logits = model(image)
    return logits

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids