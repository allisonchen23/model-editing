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

    all_data = {}
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
        # Obtain image/features/logits of anchor image
        if anchor_image is not None:
            anchor_image = torch.unsqueeze(anchor_image, dim=0)
            anchor_image = anchor_image.to(device)
            print(anchor_image.shape)
            if 'images' in data_types:
                # all_data.append(anchor_image)
                # anchor_image =
                anchor_data['images'] = anchor_image.reshape([1, -1]).cpu().numpy()
            if 'logits' in data_types or 'features' in data_types:
                print(anchor_image.shape)
                logits = context_model(anchor_image)

                if 'logits' in data_types:
                    logts = logits.reshape([1, -1])
                    anchor_data['logits'] = logits.cpu().numpy()

                if 'features' in data_types:
                    features = model.get_feature_values()
                    post_features = features['post']
                    post_features = post_features.reshape([1, -1])
                    anchor_data['features'] = post_features.cpu().numpy()


        # Obtain images/features/logits from dataloader
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
                all_images.append(image)

            # Check if we only want the image
            if 'images' in data_types and len(data_types) == 1:
                continue

            # If not image, forward it through the model
            image = image.to(device)
            logits = context_model(image)

            if 'logits' in data_types:
                all_logits.append(logits)

            if 'features' in data_types:
                features = model.get_feature_values()
                post_features = features['post']
                all_features.append(post_features)

    # For each data type,
    #   1. Concatenate
    #   2. Reshape to 1-D vectors
    #   3. Convert features/logits/images to numpy
    #   4. Add to dictionary

    if 'images' in data_types:
        all_images = torch.cat(all_images, dim=0)
        all_images = all_images.reshape([all_images.shape[0], -1])
        all_images = all_images.cpu().numpy()
        all_data['images'] = all_images

    if 'features' in data_types:
        all_features = torch.cat(all_features, dim=0)
        all_features = all_features.reshape([all_features.shape[0], -1])
        all_features = all_features.cpu().numpy()
        all_data['features'] = all_features

    if 'logits' in data_types:
        all_logits = torch.cat(all_logits, dim=0)
        all_logits = all_logits.reshape([all_logits.shape[0], -1])
        all_logits = all_logits.cpu().numpy()
        all_data['logits'] = all_logits

    # Concatenate labels
    labels = np.concatenate(labels, axis=0)

    if anchor_image is None:
        return all_data, labels, image_paths

    else:
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
    # Initialize KNN
    KNN = KNeighborsClassifier(n_neighbors=K)
    # Fit KNN to data
    KNN = KNN.fit(data, labels)
    # Obtain neighbors and respective distances to anchor
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
    all_data, all_labels, all_image_paths, all_anchor_data = _prepare_knn(
        data_loader=data_loader,
        model=model,
        anchor_image=anchor_image,
        data_types=data_types)

    output = {}
    # Obtain K nearest neighbors for each data type
    for data_type in data_types:

        # Obtain data (images, features, or logits)
        data = all_data[data_type]
        anchor_data = all_anchor_data[data_type]

        # Calculate the K nearest neighbors for the anchor
        distances, indices = _get_k_nearest_neighbors(
            K=K,
            data=data,
            labels=all_labels,
            point=anchor_data)

        # Necessary bc return values are wrapped in extra list
        indices = indices[0]
        distances = distances[0]

        # Obtain the corresponding image paths and labels
        image_paths = [all_image_paths[idx] for idx in indices]
        labels = [all_labels[idx] for idx in indices]

        # Store in dictionary
        data_type_output = {
            'indices': indices,
            'distances': distances,
            'image_paths': image_paths,
            'labels': labels
        }

        # Add to dictionary indexed by data type
        output[data_type] = data_type_output

    return output
