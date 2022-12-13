import torch
import numpy as np
import os, sys
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from scipy.spatial import distance
from PIL import Image

sys.path.insert(0, 'src/utils')
import utils
import visualizations

def _prepare_knn(data_loader, model, anchor_image=None, data_types=['features'], device=None):
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
    if model.training:
        model.eval()
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
            # Ensure 4D shape
            if len(anchor_image.shape) == 3:
                anchor_image = torch.unsqueeze(anchor_image, dim=0)
            # Move to device if applicable
            if device is not None:
                anchor_image = anchor_image.to(device)

            if 'images' in data_types:
                anchor_data['images'] = anchor_image.reshape([anchor_image.shape[0], -1]).cpu().numpy()

            # Pass image through the model
            if 'logits' in data_types or 'features' in data_types:
                logits = context_model(anchor_image)

                if 'logits' in data_types:
                    logits = logits.reshape([1, -1])
                    anchor_data['logits'] = logits.cpu().numpy()

                if 'features' in data_types:
                    features = model.get_feature_values()
                    post_features = features['post']
                    post_features = post_features.reshape([anchor_image.shape[0], -1])
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
    KNN = NearestNeighbors(n_neighbors=K)
    # Fit KNN to data
    KNN = KNN.fit(data)

    # Obtain neighbors and respective distances to anchor
    indices, distances = KNN.kneighbors(point)

    return indices, distances

def knn(K,
        data_loader,
        model,
        anchor_image,
        data_types=['features'],
        device=None,
        save_path=None):
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
        save_path : str or None
            if not None, save the dictionary as a torch checkpoint

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

    # Ensure anchor_image is a torch.tensor
    if not torch.is_tensor(anchor_image):
        anchor_image = torch.tensor(anchor_image).type(torch.float32)
    # Obtain feature representations or logits
    all_data, all_labels, all_image_paths, all_anchor_data = _prepare_knn(
        data_loader=data_loader,
        model=model,
        anchor_image=anchor_image,
        data_types=data_types,
        device=device)

    output = {}
    # Obtain K nearest neighbors for each data type
    for data_type in data_types:

        # Obtain data (images, features, or logits)
        data = all_data[data_type]
        anchor_data = all_anchor_data[data_type]

        # Calculate the K nearest neighbors for the anchor
        # print("data type: {}".format(data_type))
        # print("Data shape: {}".format(data.shape))
        # print("anchor images shape: {}".format(anchor_data.shape))
        distances, indices = _get_k_nearest_neighbors(
            K=K,
            data=data,
            labels=all_labels,
            point=anchor_data)

        # Necessary bc return values are wrapped in extra list
        # indices = indices[0]
        # distances = distances[0]

        # Obtain the corresponding image paths and labels
        image_paths = []
        labels = []
        n_points = indices.shape[0]
        for point_idx in range(n_points):
            point_image_paths = [all_image_paths[idx] for idx in indices[point_idx]]
            image_paths.append(point_image_paths)

            point_labels = [all_labels[idx] for idx in indices[point_idx]]
            labels.append(point_labels)

        # image_paths = [all_image_paths[idx] for idx in indices]
        # labels = [all_labels[idx] for idx in indices]

        # Store in dictionary
        data_type_output = {
            'indices': indices,
            'distances': distances,
            'image_paths': image_paths,
            'labels': labels
        }

        # Add to dictionary indexed by data type
        output[data_type] = data_type_output

    # Save dictionary
    if save_path is not None:
        torch.save(output, save_path)

    return output

def display_nearest_neighbors(image_paths,
                              labels,
                              items_per_row=5,
                              image_size=(2.5, 2.5),
                              row_labels=None,
                              figure_title=None,
                              font_size=12,
                              save_path=None):
    '''
    Show images of nearest neighbors

    Arg(s):
        image_paths : list[str]
            list of paths to images
        labels : list[str]
            list of labels of images

    '''
    assert len(image_paths) == len(labels)

    images = []
    for image_path in image_paths:
        image = utils.load_image(image_path)
        images.append(image)

    # Convert images and labels to grid
    images = visualizations.make_grid(images, items_per_row)
    labels = visualizations.make_grid(labels, items_per_row)

    visualizations.show_image_rows(
        images=images,
        image_titles=labels,
        row_labels=row_labels,
        figure_title=figure_title,
        font_size=font_size,
        save_path=save_path)


def distance(u, v, metric='minkowski'):
    if metric == 'minkowski':
        return distance.minkowski(u, v)
    else:
        raise ValueError("Distance metric {} not supported.".format(metric))