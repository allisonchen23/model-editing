import torch
import numpy as np
import os, sys
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from PIL import Image

sys.path.insert(0, 'src')
import utils
from utils.model_utils import quick_predict
import utils.visualizations as visualizations
import model.metric as module_metrics

def _run_model(data_loader,
               model,
               anchor_image=None,
               data_types=['features'],
               device=None):
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

    all_logits = []  # always store logits

    anchor_data = {}
    image_paths = []
    labels = []
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
            logits = context_model(anchor_image)

            # Reshape logits, convert to numpy, and store
            logits = logits.reshape([anchor_image.shape[0], -1])
            anchor_data['logits'] = logits.cpu().numpy()

            if 'features' in data_types:
                features = model.get_feature_values()
                post_features = features['post']
                post_features = post_features.reshape([anchor_image.shape[0], -1])
                anchor_data['features'] = post_features.cpu().numpy()

        # Obtain images/features/logits from dataloader
        for idx, item in enumerate(tqdm(data_loader)):

            if len(item) == 3:
                image, label, path = item
                # Add label and path to lists
                path = list(path)
                image_paths += path
                labels.append(np.asarray(label))
            else:
                image, label = item
                labels.append(np.asarray(label))

            # If we only want images, don't bother running model
            if 'images' in data_types:
                all_images.append(image)

            # If not image, forward it through the model
            image = image.to(device)
            logits = context_model(image)

            # Always append logits
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

    # if 'logits' in data_types:
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
    distances, indices = KNN.kneighbors(point)

    return distances, indices

def knn(K,
        data_loader,
        model,
        anchor_image,
        data_types=['features'],
        metric_fns=[],
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
        metric_fns : list[modules]
            list of metric functions to calculate
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

    # Create data structure to store results
    output = {}

    # Ensure anchor_image is a torch.tensor
    if not torch.is_tensor(anchor_image):
        anchor_image = torch.tensor(anchor_image).type(torch.float32)
    # Obtain feature representations or logits
    all_data, all_labels, all_image_paths, all_anchor_data = _run_model(
        data_loader=data_loader,
        model=model,
        anchor_image=anchor_image,
        data_types=data_types,
        device=device)
    # Store separately to obtain model predictions
    all_logits = all_data['logits']

    # Obtain predictions
    predictions = np.argmax(all_logits, axis=1)

    # Calculate metrics listed in metric_fns
    metrics = module_metrics.compute_metrics(
        metric_fns=metric_fns,
        prediction=predictions,
        target=all_labels,
        save_mean=True)
    output['metrics'] = metrics

    # Obtain K nearest neighbors for each data type
    knn_output = {}
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

        # Obtain the corresponding image paths, labels, and predictions
        image_paths = []
        labels = []
        predictions = []
        n_points = indices.shape[0]
        for point_idx in range(n_points):

            point_image_paths = [all_image_paths[idx] for idx in indices[point_idx]]
            image_paths.append(point_image_paths)

            # Save ground truth labels
            point_labels = np.array([all_labels[idx] for idx in indices[point_idx]])
            labels.append(point_labels)

            # Save model predictions
            point_logits = [all_logits[idx] for idx in indices[point_idx]]
            point_predictions = np.argmax(point_logits, axis=1)
            predictions.append(point_predictions)

        # Store in dictionary
        neighbor_data = data[indices]
        data_type_output = {
            'indices': indices,
            'distances': distances,
            'image_paths': image_paths,
            'labels': labels,
            'predictions': predictions,
            'anchor_data': anchor_data,
            'neighbor_data': neighbor_data
        }

        # Add to dictionary indexed by data type
        knn_output[data_type] = data_type_output

    # Add KNN output to output
    output['knn'] = knn_output
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


def calculate_distance(u, v, metric='minkowski'):
    if metric == 'minkowski':
        return distance.minkowski(u, v)
    else:
        raise ValueError("Distance metric {} not supported.".format(metric))


def calculate_distances(
    vectors,
    anchor,
    metric='minkowski'):
    '''
    Given a list of vectors, calculate the distances from anchor using metric provided

    Arg(s):
        vectors : N x D np.array or list[np.array]
            N vectors of same shape of anchor
        anchor : D-dim np.vector
            Point to calculate distance from
        metric : str
            type of distance metric to use

    Returns:
        N-dim np.array : list of distances from anchor point
    '''
    distances = []
    for vector in vectors:
        distance = calculate_distance(vector, anchor, metric=metric)
        distances.append(distance)

    distances = np.stack(distances, axis=0)

    return distances

def prediction_changes(image_paths,
                       class_list,
                       labels,
                       target,
                       predictions=None,
                       model=None,
                       device=None):
    '''
    Examine labels and predictions of subset

    Arg(s):
        image_paths : list[str]
            list of paths to images
        class_list : list[str]
            C-length list with element being corresponding class name
        target : int
            index of the target class
        labels : N-length np.array
            ground truth labels of images at image_paths
        predictions : N-length np.array
            original predictions
        model :
            new model if want to calculate new predictions
    '''
    bar_graph_data = []
    group_names = []
    n_classes = len(class_list)
    n_predictions = labels.shape[0]

    label_bins = np.bincount(labels, minlength=n_classes)

    bar_graph_data.append(label_bins)
    group_names.append('Ground Truth')

    if predictions is not None:
        prediction_bins = np.bincount(predictions, minlength=n_classes)

        bar_graph_data.append(prediction_bins)
        group_names.append('Orig. Pred.')

    print("label bins ({}): {}".format(label_bins.shape[0], label_bins))
    print("prediction bins ({}): {}".format(prediction_bins.shape[0], prediction_bins))
    if model is not None:
        assert device is not None

        logits = quick_predict(
            model=model,
            image_path=image_paths,
            device=device)

        model_predictions = torch.argmax(logits, dim=1)
        model_predictions = model_predictions.cpu().numpy()
        model_prediction_bins = np.bincount(model_predictions, minlength=n_classes)

        bar_graph_data.append(model_prediction_bins)
        group_names.append('Edited Pred.')
        print("new prediction bins: {}".format(model_prediction_bins.shape[0], model_prediction_bins))

    bar_graph_data = np.stack(bar_graph_data, axis=0)

    visualizations.bar_graph(
        data=bar_graph_data,
        labels=class_list,
        groups=group_names)

    # Calculate % of predictions that changed to target
    n_changed_to_target = np.sum(np.where(((predictions != target) & (model_predictions == target)), 1, 0))
    n_unaffected = np.sum(np.where(model_predictions == predictions, 1, 0))

    results = {}
    results['original_distribution'] = prediction_bins
    results['edited_distribution'] = model_prediction_bins
    results['n_changed_to_target'] = n_changed_to_target
    results['n_unaffected'] = n_unaffected

    return results