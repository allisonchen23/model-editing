import torch
import numpy as np
import os, sys
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from PIL import Image

sys.path.insert(0, 'src')
import utils
from utils import load_image, informal_log, read_json, read_lists
from parse_config import ConfigParser
from utils.model_utils import quick_predict, prepare_device
import utils.visualizations as visualizations
import model.metric as module_metrics
import model.model as module_arch

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
        image = load_image(image_path)
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

def predict_and_compare(image_paths,
                       class_list,
                       labels,
                       target,
                       predictions=None,
                       model=None,
                       bar_plot_save_path=None,
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
        model : torch.nn.module
            new model if want to calculate new predictions
        bar_plot_save_path : str or None
            optional path to save bar plots to
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

    # print("label bins ({}): {}".format(label_bins.shape[0], label_bins))
    # print("prediction bins ({}): {}".format(prediction_bins.shape[0], prediction_bins))
    if model is not None:
        assert device is not None

        logits = quick_predict(
            model=model,
            image=image_paths,
            device=device)

        model_predictions = torch.argmax(logits, dim=1)
        model_predictions = model_predictions.cpu().numpy()
        model_prediction_bins = np.bincount(model_predictions, minlength=n_classes)

        bar_graph_data.append(model_prediction_bins)
        group_names.append('Edited Pred.')
        # print("new prediction bins: {}".format(model_prediction_bins.shape[0], model_prediction_bins))

    bar_graph_data = np.stack(bar_graph_data, axis=0)

    visualizations.bar_graph(
        data=bar_graph_data,
        labels=class_list,
        groups=group_names,
        save_path=bar_plot_save_path)

    # Calculate % of predictions that changed to target
    n_changed_to_target = np.sum(np.where(((predictions != target) & (model_predictions == target)), 1, 0))
    n_unaffected = np.sum(np.where(model_predictions == predictions, 1, 0))

    results = {}
    results['original_distribution'] = prediction_bins
    results['edited_distribution'] = model_prediction_bins
    results['n_changed_to_target'] = n_changed_to_target
    results['n_unaffected'] = n_unaffected

    return results

def analyze_prediction_changes(pre_edit_knn,
                               post_edit_logits,
                               model,
                               class_list,
                               target_class_idx,
                               device,
                               visualizations_dir=None):
    '''
    Analyze how predictions change in the pre-edit neighbors of key and value (features and logits)

    Arg(s):
        pre_edit_knn : dict
            dictionary of data saved from knn() in knn.py from pre-edit
        post_edit_logits : dict
            logits of neighbors and anchor data after edit
        model : torch.nn.module
            model to make new predictions on pre-edit neighbors
        class_list : list[str]
            list of class names
        target_class_idx : int
            the target class of key image
        device : torch.device
            device to run model predictions on
        visualizations_dir : str or None
            (opt) directory to store bar graph visualizations

    Returns:
        results : dict
            results of
                * prediction changes of all 4 (key/val x features/logits)'s neighbors
                * pre/post predictions of anchor images
    '''
    results = {}
    for data_type in ['features', 'logits']:
        for anchor_type in [0, 1]:  # 0: key, 1: value
            data_anchor_id = "{}_{}".format(data_type, 'key' if anchor_type==0 else 'value')

            # Obtain pre-edit neighbor images, true labels, and original predictions
            image_paths = pre_edit_knn[data_type]['image_paths'][anchor_type]
            labels = pre_edit_knn[data_type]['labels'][anchor_type]
            original_predictions = pre_edit_knn[data_type]['predictions'][anchor_type]

            if visualizations_dir is None:
                bar_plot_save_path = None
            else:
                bar_plot_save_path = os.path.join(visualizations_dir, "{}_bar_plot.png".format(data_anchor_id))

            compared_outputs = predict_and_compare(
                image_paths=image_paths,
                labels=labels,
                class_list=class_list,
                target=target_class_idx,
                predictions=original_predictions,
                model=model,
                bar_plot_save_path=bar_plot_save_path,
                device=device)

            results[data_anchor_id] = compared_outputs

    # Store original predictions for key and value
    pre_edit_key_logits = pre_edit_knn['logits']['anchor_data'][0]
    pre_edit_val_logits = pre_edit_knn['logits']['anchor_data'][1]
    results['pre_key_prediction'] = np.argmax(pre_edit_key_logits)
    results['pre_val_prediction'] = np.argmax(pre_edit_val_logits)
    # Store edited predictions for key and value
    post_edit_key_logits = post_edit_logits['anchor_data'][0]
    post_edit_value_logits = post_edit_logits['anchor_data'][1]
    results['post_key_prediction'] = np.argmax(post_edit_key_logits)
    results['post_val_prediction'] = np.argmax(post_edit_value_logits)

    return results

def analyze_distances(data_type,
                      pre_edit_values,
                      post_edit_values,
                      model,
                      device):

    # For easy access in dict keys
    keywords = ['key', 'val']

    # Extract values
    pre_edit_key_values = pre_edit_values['anchor_data'][0]
    pre_edit_val_values = pre_edit_values['anchor_data'][1]
    post_edit_key_values = post_edit_values['anchor_data'][0]
    post_edit_val_values = post_edit_values['anchor_data'][1]

    distance_results = {}

    # Calculate distance between key -> val before and after edit
    pre_edit_key_val_distance = calculate_distance(pre_edit_key_values, pre_edit_val_values)
    post_edit_key_val_distance = calculate_distance(post_edit_key_values, post_edit_val_values)
    distance_results['key_val'] = (pre_edit_key_val_distance, post_edit_key_val_distance)

    for anchor in [0, 1]: # 0/1 refers to key/val
        # Obtain features/logits for anchor
        pre_edit_anchor_values = pre_edit_values['anchor_data'][anchor]
        post_edit_anchor_values = post_edit_values['anchor_data'][anchor]
        for neighbors in [0, 1]:  # 0/1 refers to key/val

            # Obtain pre-edit feature/logits for neighbors
            pre_edit_neighbor_values = pre_edit_values['neighbor_data'][neighbors]
            # Obtain post-edit features/logits for neighbors
            pre_edit_neighbor_paths = pre_edit_values['image_paths'][neighbors]
            neighbor_edited_logits = quick_predict(
                model=model,
                image=pre_edit_neighbor_paths,
                device=device)
            if data_type == 'features':
                neighbor_edited_features = model.get_feature_values()['post']
                neighbor_edited_features = neighbor_edited_features.reshape(
                    [neighbor_edited_features.shape[0], -1])
                post_edit_neighbor_values = neighbor_edited_features.cpu().numpy()
            else:
                post_edit_neighbor_values = neighbor_edited_logits.cpu().numpy()

            # Calculate mean distance pre-edit
            mean_pre_edit_distance = np.mean(
                calculate_distances(
                vectors=pre_edit_neighbor_values,
                anchor=pre_edit_anchor_values))

            # Calculate mean distance post edit
            mean_post_edit_distance = np.mean(
                calculate_distances(
                vectors=post_edit_neighbor_values,
                anchor=post_edit_anchor_values))

            # Store in dictionary
            dict_key = "{}_{}N".format(keywords[anchor], keywords[neighbors])
            distance_results[dict_key] = (mean_pre_edit_distance, mean_post_edit_distance)
            # print(dict_key, mean_pre_edit_distance, mean_post_edit_distance)

    return distance_results


def load_and_analyze_knn(restore_dir,
                         pre_edit_knn_path,
                         post_edit_knn_path,
                         knn_analysis_filename,
                         target_class_idx,
                         class_list,
                         progress_report_path=None,
                         # knn_data_types=['images', 'features', 'logits'],
                         save_images=False,
                         save_plots=True):
    '''
    Given where KNN results are stored, analyze them to calculate changes in predictions and distances with edit
    Saves results in restore_dir

    Arg(s):
        restore_dir : str
            directory where everything is saved
        pre_edit_knn_path : str
            path to pre-edit knn results
        post_edit_knn_path : str
            path to post-edit knn results
        knn_analysis_filename : str
            name of file to save results to
        target_class_idx : int
            index of target class
        class_list : list[str]
            list of class names
        progress_report_path : str or None
            (opt) path to the progress log
        save_images : bool
            whether or not to save neighbor visualizations (not supported currently)
        save_plots : bool
            whether or not to save bar plots

    Returns:
        None
    '''

    informal_log("Analyzing KNN results from {}".format(restore_dir), progress_report_path)
    # Create paths to save results
    visualizations_dir = os.path.join(restore_dir, 'knn_visualizations')
    log_path = os.path.join(visualizations_dir, "knn_analysis_log.txt")
    if os.path.exists(log_path):
        os.remove(log_path)

    save_results_path = os.path.join(restore_dir, knn_analysis_filename)
    informal_log("Logging and saving visualizations to {}".format(log_path), progress_report_path)
    informal_log("Saving results to {}".format(save_results_path), progress_report_path)

    # Load config file
    config_path = os.path.join(restore_dir, "config.json")
    if not os.path.exists(config_path):
        raise ValueError("Config file at {} does not exist.".format(config_path))
    config_json = read_json(config_path)
    config = ConfigParser(config_json, make_dirs=False)

    # Extract information from config file
    K = config_json['editor']['K']
    layernum = config_json['layernum']
    device, device_ids = prepare_device(config['n_gpu'])

    # Load edited model
    edited_model_path = os.path.join(restore_dir, "edited_model.pth")
    edited_model = config.init_obj('arch', module_arch, layernum=layernum)
    edited_model.restore_model(edited_model_path)
    # edited_context_model = edited_model.context_model
    edited_model.eval()
    informal_log("Restored edited model from {}".format(edited_model_path), progress_report_path)

    # Load KNN results
    pre_edit_knn = torch.load(pre_edit_knn_path)
    post_edit_knn = torch.load(post_edit_knn_path)
    informal_log("Loaded pre-edit KNN results from {}.".format(pre_edit_knn_path), progress_report_path)
    informal_log("Loaded post-edit KNN results from {}.".format(post_edit_knn_path), progress_report_path)

    # KNN Analysis results data structure
    knn_analysis_results = {}

    # Calculate change in predictions first
    informal_log("Analyzing prediction changes...")
    prediction_changes_results = analyze_prediction_changes(
        pre_edit_knn=pre_edit_knn,
        post_edit_logits=post_edit_knn['logits'],
        model=edited_model,
        class_list=class_list,
        target_class_idx=target_class_idx,
        device=device,
        visualizations_dir=visualizations_dir if save_plots else None)
    # Add to results dictionary
    knn_analysis_results['prediction_changes'] = prediction_changes_results

    # print(prediction_changes_results)

    # Analyze changes in distances for both features and logits
    distance_results = {}

    for data_type in ['features', 'logits']:
        pre_edit_values = pre_edit_knn[data_type]
        post_edit_values = post_edit_knn[data_type]

        value_distances = analyze_distances(
            data_type=data_type,
            pre_edit_values=pre_edit_values,
            post_edit_values=post_edit_values,
            model=edited_model,
            device=device)
        distance_results[data_type] = value_distances
    knn_analysis_results['distance_results'] = distance_results

    torch.save(knn_analysis_results, save_results_path)
    informal_log("Saved KNN analysis results to {}".format(save_results_path), progress_report_path)

    informal_log("", progress_report_path)

def analyze_knn(save_dir,
                config,
                pre_edit_knn,
                post_edit_knn,
                edited_model,
                knn_analysis_filename,
                target_class_idx,
                class_list,
                progress_report_path=None,
                # knn_data_types=['images', 'features', 'logits'],
                save_images=False,
                save_plots=True):
    '''
    Given where KNN results are stored, analyze them to calculate changes in predictions and distances with edit
    Saves results in restore_dir

    Arg(s):
        restore_dir : str
            directory where everything is saved
        pre_edit_knn_path : str
            path to pre-edit knn results
        post_edit_knn_path : str
            path to post-edit knn results
        knn_analysis_filename : str
            name of file to save results to
        target_class_idx : int
            index of target class
        class_list : list[str]
            list of class names
        progress_report_path : str or None
            (opt) path to the progress log
        save_images : bool
            whether or not to save neighbor visualizations (not supported currently)
        save_plots : bool
            whether or not to save bar plots

    Returns:
        None
    '''

    informal_log("Saving KNN analysis results to {}".format(save_dir), progress_report_path)
    # Create paths to save results
    visualizations_dir = os.path.join(save_dir, 'knn_visualizations')
    log_path = os.path.join(visualizations_dir, "knn_analysis_log.txt")
    if os.path.exists(log_path):
        os.remove(log_path)

    save_results_path = os.path.join(save_dir, knn_analysis_filename)
    informal_log("Logging and saving visualizations to {}".format(log_path), progress_report_path)
    informal_log("Saving results to {}".format(save_results_path), progress_report_path)

    # Load config file
    # config_path = os.path.join(save_dir, "config.json")
    # if not os.path.exists(config_path):
    #     raise ValueError("Config file at {} does not exist.".format(config_path))
    # config_json = read_json(config_path)
    # config = ConfigParser(config_json, make_dirs=False)

    # Extract information from config file
    K = config.config['editor']['K']
    layernum = config.config['layernum']
    device, device_ids = prepare_device(config.config['n_gpu'])

    # Load edited model
    # edited_model_path = os.path.join(restore_dir, "edited_model.pth")
    # edited_model = config.init_obj('arch', module_arch, layernum=layernum)
    # edited_model.restore_model(edited_model_path)
    # edited_context_model = edited_model.context_model
    edited_model.eval()
    # informal_log("Restored edited model from {}".format(edited_model_path), progress_report_path)

    # Load KNN results
    # pre_edit_knn = torch.load(pre_edit_knn_path)
    # post_edit_knn = torch.load(post_edit_knn_path)
    # informal_log("Loaded pre-edit KNN results from {}.".format(pre_edit_knn_path), progress_report_path)
    # informal_log("Loaded post-edit KNN results from {}.".format(post_edit_knn_path), progress_report_path)

    # KNN Analysis results data structure
    knn_analysis_results = {}

    # Calculate change in predictions first
    informal_log("Analyzing prediction changes...")
    prediction_changes_results = analyze_prediction_changes(
        pre_edit_knn=pre_edit_knn,
        post_edit_logits=post_edit_knn['logits'],
        model=edited_model,
        class_list=class_list,
        target_class_idx=target_class_idx,
        device=device,
        visualizations_dir=visualizations_dir if save_plots else None)
    # Add to results dictionary
    knn_analysis_results['prediction_changes'] = prediction_changes_results

    # print(prediction_changes_results)

    # Analyze changes in distances for both features and logits
    distance_results = {}

    for data_type in ['features', 'logits']:
        pre_edit_values = pre_edit_knn[data_type]
        post_edit_values = post_edit_knn[data_type]

        value_distances = analyze_distances(
            data_type=data_type,
            pre_edit_values=pre_edit_values,
            post_edit_values=post_edit_values,
            model=edited_model,
            device=device)
        distance_results[data_type] = value_distances
    knn_analysis_results['distance_results'] = distance_results

    torch.save(knn_analysis_results, save_results_path)
    informal_log("Saved KNN analysis results to {}".format(save_results_path), progress_report_path)

    informal_log("", progress_report_path)
