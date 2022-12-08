import torch
import numpy as np
import os, sys
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

def _prepare_knn(data_loader, model, anchor_image=None, data_type='features'):
    '''
    Obtain nearest neighbors for each image in data loader and base image (if not None)

    Arg(s):
        K : int
            how many neighbors to calculate
        data_loader : torch.utils.DataLoader
            shuffle should be false
        model : torch.nn.module
            model
        anchor_image : torch.tensor or None
            specific image to calculate neighbors for
        data_type : str
            for what data we want to calculate KNN for -- features, logits, images
    '''
    assert data_type in ['features', 'logits', 'images'], "Unsupported data type {}".format(data_type)
    assert not model.training
    assert model.__class__.__name__ == 'CIFAR10PretrainedModelEdit'

    all_data = []
    image_paths = []
    labels = []
    return_paths = data_loader.get_return_paths()
    context_model = model.context_model

    with torch.no_grad():
        # First element in all_data will be the anchor_image representation if it's not None
        if anchor_image is not None:
            anchor_image = torch.unsqueeze(anchor_image, dim=0)
            anchor_image = anchor_image.to(device)
            if data_type == 'images':
                # all_data.append(anchor_image)
                anchor_data = anchor_image
            else:
                logits = context_model(anchor_image)

                if data_type == 'logits':
                    anchor_data = logits
                elif data_type == 'features':
                    features = model.get_feature_values()
                    post_features = features['post']
                    anchor_data = post_features

            # Flatten to a 1-D vector
            anchor_data = anchor_data.reshape([1, -1])

        # Obtain features from dataset
        for idx, item in enumerate(tqdm(data_loader)):
            if return_paths:
                image, label, path = item
                # Add label and path to lists
                path = list(path)
                image_paths += path
                labels.append(np.asarray(label))
            else:
                image, label = item


            # If we only want images, don't bother running model
            if data_type == 'images':
                all_data.append(image)
                continue

            # If not image, forward it through the model
            image = image.to(device)
            logits = context_model(image)

            if data_type == 'logits':
                all_data.append(logits)
            elif data_type == 'features':
                features = model.get_feature_values()
                post_features = features['post']

                print("post_features.shape {}".format(post_features.shape))
                all_data.append(post_features)

    # Concatenate, reshape to 1-D vectors, and convert features/logits/images to numpy
    all_data = torch.cat(all_data, dim=0)
    all_data = all_data.reshape([all_data.shape[0], -1])
    all_data = all_data.cpu().numpy()
    assert len(all_data.shape) == 2

    # Concatenate labels
    labels = np.concatenate(labels, axis=0)

    if anchor_image is None:
        return all_data, labels, image_paths

    else:
        anchor_data = anchor_data.cpu().numpy()
        return all_data, labels, image_paths, anchor_data

def _get_k_nearest_neighbors(K, data, labels, point):
    '''
    Given a data point and data, return the indices of the K nearest neighbors

    Arg(s):
        K : int
            number of neighbors to return
        data : N x ... np.array
            data from dataset to find neighbors from
        point : 1 x ... np.array
            point to find all neighbors for, same shape except in dim=0 as data

    Returns:
        tuple(list[int], list[float])
            tuple of 2 K length lists of indices and corresponding distances
                from data that correspond with nearest neighbors to point
    '''
    KNN = KNeighborsClassifier(n_neighbors=K)
    KNN = KNN.fit(data, labels)
    indices, distances = KNN.kneighbors(point)

    return indices, distances

def knn(K, data_loader, model, anchor_image, data_type='features'):
    '''
    Given a base image and a dataset, find the K nearest neighbors according to model

    Arg(s):
        K : int
            number of neighbors to return
        data_loader : torch.utils.data.DataLoader
            Data loader to obtain neighbors from
        model : torch.nn.Module
            model to obtain features/predictions from
        anchor_image : np.array
            image of which we want to find neighbors for
        data_type : str
            choice of ['image', 'features', 'logits']
            where features is directly after the edited layer

    Returns:
        tuple(list[str], list[int])
            list of paths to original images and labels of the K nearest neighbors
    '''
    if not data_loader.get_return_paths():
        raise ValueError("DataLoader must return paths.")

    # Obtain feature representations or logits
    data, labels, image_paths, anchor_data = prepare_knn(
        data_loader=data_loader,
        model=model,
        anchor_image=anchor_image,
        data_type=data_type)

    # Obtain K nearest neighbors
    distances, indices = _get_k_nearest_neighbors(
        K=K,
        data=data,
        point=anchor_data)

    # Return values are wrapped in extra list
    distances = distances[0]
    indices = indices[0]

    # Obtain the corresponding image paths and labels
    neighbor_image_paths = [image_paths[idx] for idx in indices[0]]
    neighbor_labels = [labels[idx] for idx in indices[0]]

    return neighbor_image_paths, neighbor_labels
