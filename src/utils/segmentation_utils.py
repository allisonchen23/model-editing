import numpy as np
import skimage.segmentation as segmentation
import skimage.filters as filters
import skimage.color as color

def segment(image, method, params):
    '''
    Given an image and a segmentation method, and necessary keyword arguments, return segments

    Arg(s):
        image : C x H x W np.array
            Image to segment
        method : str
            type of segmentation method
        params : dict
            keyword arguments for specific segmentation method

    Returns:
        segments : H x W np.array
            Each segment labeled by integer index
    '''

    # Assert image is in C x H x W shape
    assert len(image.shape) == 3, "Image expected to have shape C x H x W"
    assert image.shape[0] == 3, "Image expected to have 3 channels in dimension 0"
    params['channel_axis'] = 0

    if method == 'felzenszwalb':
        segments = segmentation.felzenszwalb(
            image,
            **params)
    elif method == 'quickshift':
        segments = segmentation.quickshift(
            image,
            **params)
    elif method == 'slic':
        segments = segmentation.slic(
            image,
            **params)

    elif method == 'watershed':
        del params['channel_axis']
        gradient = filters.sobel(color.rgb2gray(np.transpose(image, axes=[1, 2, 0])))
        segments = segmentation.watershed(
            gradient,
            **params)
    else:
        raise ValueError("Segmentation method {} not supported.".format(method))
    return segments