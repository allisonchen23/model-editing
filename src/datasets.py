import torchvision
import os
import torchvision.transforms as transforms

# cinic_directory = '/path/to/cinic/directory'
# cinic_mean = [0.47889522, 0.47227842, 0.43047404]
# cinic_std = [0.24205776, 0.23828046, 0.25874835]
# cinic_train = torch.utils.data.DataLoader(
#     torchvision.datasets.ImageFolder(cinic_directory + '/train',
#     	transform=transforms.Compose([transforms.ToTensor(),
#         transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
#     batch_size=128, shuffle=True)

def get_dataset(dataset_path,
                split='train',
                normalize=False,
                mean=None,
                std=None):
    '''
    Return the respective torchvision.datasets dataset
    
    Arg(s):
        split : str
            choose from ['train', 'test', 'valid']
        normalize : bool
            whether to normalize or not
        mean : list[float] or None
            if normalize, must not be None.
            Mean values of dataset for RGB values
        std : list[float] or None
            if normalize, must not be None.
            Standard deviation values of dataset for RGB values
        
    Returns torchvision.dataset
    '''
    
    assert split in ['train', 'test', 'valid']
    dataset_path = os.path.join(dataset_path, split)
    transform_operations = [transforms.ToTensor()]
    if normalize:
        assert mean is not None and len(mean) == 3
        assert std is not None and len(std) == 3
        transform_operations.append(transforms.Normalize(mean=mean, std=std))
        
    return torchvision.datasets.ImageFolder(
        dataset_path,
        transform=transforms.Compose(transform_operations))
    
    