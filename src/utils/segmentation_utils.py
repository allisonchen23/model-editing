import numpy as np
import skimage.segmentation as segmentation
import skimage.filters as filters
import skimage.color as color

def segment(image, method, kwargs):
    '''
    Given an image and a segmentation method, and necessary keyword arguments, return segments

    Arg(s):
        image : C x H x W np.array
            Image to segment
        method : str
            type of segmentation method
        kwargs : dict
            keyword arguments for specific segmentation method

    Returns:
        segments : H x W np.array
            Each segment labeled by integer index
    '''

    # Assert image is in C x H x W shape
    assert len(image.shape) == 3, "Image expected to have shape C x H x W"
    assert image.shape[0] == 3, "Image expected to have 3 channels in dimension 0"
    kwargs['channel_axis'] = 0

    if method == 'felzenszwalb':
        segments = segmentation.felzenszwalb(
            image,
            **kwargs)
    elif method == 'quickshift':
        segments = segmentation.quickshift(
            image,
            **kwargs)
    elif method == 'slic':
        segments = segmentation.slic(
            image,
            **kwargs)

    elif method == 'watershed':
        del kwargs['channel_axis']
        gradient = filters.sobel(color.rgb2gray(np.transpose(image, axes=[1, 2, 0])))
        segments = segmentation.watershed(
            gradient,
            **kwargs)
    else:
        raise ValueError("Segmentation method {} not supported.".format(method))
    return segments

'''
Modification Methods
'''
def mask_out(image, mask):
    '''
    Given an image and binary mask, black out where mask is activated

    Arg(s):
        image : C x H x W np.array
            original image
        mask : C x H x W np.array (2D array broadcasted in C dimension)
            mask

    Returns:
        image : C x H x W np.array
            image with masked portion blacked out
    '''
    image = np.where(mask == 1, 0, image)
    return image

def gaussian_noise(image, mask, mean=0, std=0.1):
    '''
    Given an image and binary mask, apply Gaussian noise to masked region

    Arg(s):
        image : C x H x W np.array
            original image
        mask : C x H x W np.array (2D array broadcasted in C dimension)
            mask
        mean : float
            mean for Gaussian distribution
        std : float
            standard deviation for Gaussian distribution

    Returns:
        image : C x H x W np.array
            image with noise added to masked portion
    '''
    normal_distribution = np.random.normal(mean, std, size=image.shape)
    image = np.where(mask == 1, image + normal_distribution, image)
    return image


def modify_segments(image,
                    segments,
                    method,
                    **kwargs):
    '''
    Given an image, the segmentation, and modification, return list of images with each segment modified

    Arg(s):
        image : C x H x W np.array (np.float)
            original image
        segments : H x W np.array (np.int)
            segmented image (e.g. output of utils.segmentation_utils.segment()
        method : str
            type of modification to perform

    Returns:
        modified_images : list[C x H x W np.array]
            list of modified images, one per segment
    '''
    print("KWargs: {}".format(kwargs))
    # Obtain number of segments
    unique_segments = np.unique(segments)
    n_segments = len(unique_segments)
    broadcasted_segments = np.broadcast_to(segments, shape=(3, *segments.shape))
    modified_images = []

    # Determine modification method
    if method == 'gaussian_noise':
        modification_func = gaussian_noise
        assert 'mean' in kwargs and 'std' in kwargs, "Gaussian noise requires keyword arguments 'mean' and 'std'"

    elif method == 'mask':
        modification_func = mask_out
        kwargs = {}
    else:
        raise ValueError("Modification method {} not supported".format(method))

    # Modify each segment
    for label in unique_segments:
        label_mask = np.where(segments == label, 1, 0)

        modified_image = modification_func(image, label_mask, **kwargs)
        modified_images.append(modified_image)


    return modified_images