import torch
import numpy as np
import os, sys
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

def _prepare_knn(data_loader, model, anchor_image=None, data_types=['features']):
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
        data_types : list[str]
            for what data we want to calculate KNN for -- features, logits, images
    '''
    for data_type in data_types:
        assert data_type in ['features', 'logits', 'images'], "Unsupported data type {}".format(data_types)
    assert not model.training
    assert model.__class__.__name__ == 'CIFAR10PretrainedModelEdit'

    # all_data = []
    if 'images' in data_types:
        all_images = []
    if 'features' in data_types:
        all_features = []
    if 'logits' in data_types:
        all_logits = []

    anchor_data = {}
    image_paths = []
    labels = []
    return_paths = data_loader.get_return_paths()
    context_model = model.context_model

    with torch.no_grad():
        # First element in all_data will be the anchor_image representation if it's not None
        if anchor_image is not None:
            anchor_image = torch.unsqueeze(anchor_image, dim=0)
            anchor_image = anchor_image.to(device)
            if 'images' in data_types:
                # all_data.append(anchor_image)
                anchor_image = anchor_image.reshape([1, -1])
                anchor_data['image'] = anchor_image.cpu().numpy()
            if 'logits' in data_types or 'features' in data_types:
                logits = context_model(anchor_image)

                if 'logits' in data_types:
                    logts = logits.reshape([1, -1])
                    anchor_data['logits'] = logits.cpu().numpy()
                elif data_types == 'features':
                    features = model.get_feature_values()
                    post_features = features['post']
                    post_features = post_features.reshape([1, -1])
                    anchor_data['features'] = post_features.cpu().numpy()


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
            if 'images' in data_types:
                images.append(image)
                continue

            # Check if we only want the image
            if 'images' in data_types and len(data_types) == 1:
                continue

            # If not image, forward it through the model
            image = image.to(device)
            logits = context_model(image)

            if 'logits' in data_types:
                logits.append(logits)
            if 'features' in data_types:
                features = model.get_feature_values()
                post_features = features['post']

                print("post_features.shape {}".format(post_features.shape))
                features.append(post_features)

    # Concatenate, reshape to 1-D vectors, and convert features/logits/images to numpy
    if 'images' in data_types:
        images = torch.cat(images, dim=0)
        images = images.reshape([images.shape[0], -1])
        images = images.cpu().numpy()
        all_data['images'] = images

    if 'features' in data_types:
        features = torch.cat(features, dim=0)
        features = images.reshape([features.shape[0], -1])
        features = features.cpu().numpy()
        all_data['features'] = features

    if 'logits' in data_types:
        logits = torch.cat(logits, dim=0)
        logits = images.reshape([logits.shape[0], -1])
        logits = logits.cpu().numpy()
        all_data['logits'] = logits
    # all_data = torch.cat(all_data, dim=0)
    # all_data = all_data.reshape([all_data.shape[0], -1])
    # all_data = all_data.cpu().numpy()
    # assert len(all_data.shape) == 2

    # Concatenate labels
    labels = np.concatenate(labels, axis=0)

    if anchor_image is None:
        return all_data, labels, image_paths

    else:
        # anchor_data = anchor_data.cpu().numpy()
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

def knn(K, data_loader, model, anchor_image, data_types='features'):
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
        data_types : str
            choice of ['image', 'features', 'logits']
            where features is directly after the edited layer

    Returns:
        tuple(
            list[int],
            list[float],
            list[str],
            list[int])

            indices, distances, image_paths, labels
    '''
    if not data_loader.get_return_paths():
        raise ValueError("DataLoader must return paths.")

    # Obtain feature representations or logits
    data, labels, image_paths, anchor_data = prepare_knn(
        data_loader=data_loader,
        model=model,
        anchor_image=anchor_image,
        data_types=data_types)

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

    return indices, distances, neighbor_image_paths, neighbor_labels
