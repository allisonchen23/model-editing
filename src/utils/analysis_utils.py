import pandas as pd
from tqdm import tqdm
import os
import sys
import torch
import numpy as np
import argparse

sys.path.insert(0, 'src')
from utils import read_lists

parser = argparse.ArgumentParser()

def combine_results(data_id,
                    knn_analysis,
                    pre_edit_metrics,
                    post_edit_metrics):
    '''
    Given list of dictionaries, combine into 1 dictionary

    Arg(s):
        knn_analysis : dict{str : any}
            dictionary of results from knn_analysis.ipynb
        pre_edit_metrics : dict{str : any}
            dictionary of pre edit metrics
        post_edit_metrics : dict{str : any}
            dictionary of post edit metrics

    Returns:
        master_dict : dict{str: any}
    '''
    # Get sub-dictionaries of knn_analysis
    prediction_changes = knn_analysis['prediction_changes']
    distances = knn_analysis['distance_results']

    # Obtain target and original class predictions

    target_class_idx = prediction_changes['pre_val_prediction']
    original_class_idx = prediction_changes['pre_key_prediction']

    master_dict = {}
    master_dict['ID'] = data_id

    # Data from metric dictionaries
    # Store Accuracy
    master_dict['Pre Accuracy'] = pre_edit_metrics['accuracy']
    master_dict['Post Accuracy'] = post_edit_metrics['accuracy']

    # Store Mean Precision
    master_dict['Pre Mean Precision'] = pre_edit_metrics['precision_mean']
    master_dict['Post Mean Precision'] = post_edit_metrics['precision_mean']

    # Store Mean Recall
    master_dict['Pre Mean Recall'] = pre_edit_metrics['recall_mean']
    master_dict['Post Mean Recall'] = post_edit_metrics['recall_mean']

    # Store Mean F1
    master_dict['Pre Mean F1'] = pre_edit_metrics['f1_mean']
    master_dict['Post Mean F1'] = post_edit_metrics['f1_mean']


    # Store Target Precision
    master_dict['Pre Target Precision'] = pre_edit_metrics['precision'][target_class_idx]
    master_dict['Post Target Precision'] = post_edit_metrics['precision'][target_class_idx]

    # Store Target Recall
    master_dict['Pre Target Recall'] = pre_edit_metrics['recall'][target_class_idx]
    master_dict['Post Target Recall'] = post_edit_metrics['recall'][target_class_idx]

    # Store Target F1
    master_dict['Pre Target F1'] = pre_edit_metrics['f1'][target_class_idx]
    master_dict['Post Target F1'] = post_edit_metrics['f1'][target_class_idx]


    # Store Original Class Precision
    master_dict['Pre Orig Pred Precision'] = pre_edit_metrics['precision'][original_class_idx]
    master_dict['Post Orig Pred Precision'] = post_edit_metrics['precision'][original_class_idx]

    # Store Original Class Recall
    master_dict['Pre Orig Pred Recall'] = pre_edit_metrics['recall'][original_class_idx]
    master_dict['Post Orig Pred Recall'] = post_edit_metrics['recall'][original_class_idx]

    # Store Original Class F1
    master_dict['Pre Orig Pred F1'] = pre_edit_metrics['f1'][original_class_idx]
    master_dict['Post Orig Pred F1'] = post_edit_metrics['f1'][original_class_idx]

    # Store class distributions pre and post edit
    master_dict['Pre Class Dist'] = pre_edit_metrics['predicted_class_distribution']
    master_dict['Post Class Dist'] = post_edit_metrics['predicted_class_distribution']

    # Data from knn analysis dictionaries
    # Predictions of key and value
    master_dict['Pre key Prediction'] = prediction_changes['pre_key_prediction']
    master_dict["Post key Prediction"] = prediction_changes['post_key_prediction']
    master_dict['Pre val Prediction'] = prediction_changes['pre_val_prediction']
    master_dict["Post val Prediction"] = prediction_changes['post_val_prediction']

    # Number of neighbors that became target
    master_dict["Num of key's Neighbors Became Target (F)"] = prediction_changes['features_key']['n_changed_to_target']
    master_dict["Num of key's Neighbors Became Target (L)"] = prediction_changes['logits_key']['n_changed_to_target']
    master_dict["Num of val's Neighbors Became Target (F)"] = prediction_changes['features_value']['n_changed_to_target']
    master_dict["Num of val's Neighbors Became Target (L)"] = prediction_changes['logits_value']['n_changed_to_target']

    master_dict["Num of key's Neighbors Unaffected (F)"] = prediction_changes['features_key']['n_unaffected']
    master_dict["Num of key's Neighbors Unaffected (L)"] = prediction_changes['logits_key']['n_unaffected']
    master_dict["Num of val's Neighbors Unaffected (F)"] = prediction_changes['features_value']['n_unaffected']
    master_dict["Num of val's Neighbors Unaffected (L)"] = prediction_changes['logits_value']['n_unaffected']

    # Examine Distances
    # Distance between key-val
    master_dict["Pre key-val (F)"] = distances['features']['key_val'][0]
    master_dict["Post key-val (F)"] = distances['features']['key_val'][1]
    master_dict["Pre key-val (L)"] = distances['logits']['key_val'][0]
    master_dict["Post key-val (L)"] = distances['logits']['key_val'][1]

    # Distance between key's neighbors -> val
    master_dict["Pre val-keyN (F)"] = distances['features']['val_keyN'][0]
    master_dict["Post val-keyN (F)"] = distances['features']['val_keyN'][1]
    master_dict["Pre val-keyN (L)"] = distances['logits']['val_keyN'][0]
    master_dict["Post val-keyN (L)"] = distances['logits']['val_keyN'][1]

    # Distances between val's neighbors and key
    master_dict["Pre key-valN (F)"] = distances['features']['key_valN'][0]
    master_dict["Post key-valN (F)"] = distances['features']['key_valN'][1]
    master_dict["Pre key-valN (L)"] = distances['logits']['key_valN'][0]
    master_dict["Post key-valN (L)"] = distances['logits']['key_valN'][1]

    # Distances between key's neighbors and key
    master_dict["Pre key-keyN (F)"] = distances['features']['key_keyN'][0]
    master_dict["Post key-keyN (F)"] = distances['features']['key_keyN'][1]
    master_dict["Pre key-keyN (L)"] = distances['logits']['key_keyN'][0]
    master_dict["Post key-keyN (L)"] = distances['logits']['key_keyN'][1]

    # Distances between val's neighbors and val
    master_dict["Pre val-valN (F)"] = distances['features']['val_valN'][0]
    master_dict["Post val-valN (F)"] = distances['features']['val_valN'][1]
    master_dict["Pre val-valN (L)"] = distances['logits']['val_valN'][0]
    master_dict["Post val-valN (L)"] = distances['logits']['val_valN'][1]

    return master_dict

def store_csv(trial_dirs,
              class_list,
              save_path):

    n_trials = len(trial_dirs)

    data = []
    for trial_idx, trial_dir in tqdm(enumerate(trial_dirs)):
        # Obtain key ID from path
        key_id = os.path.basename(os.path.dirname(trial_dir))
        id_class = key_id.split('-')[0]
        if id_class not in class_list:
            raise ValueError("Invalid key_id {}".format(key_id))

        # Obtain value ID from path
        val_id = os.path.basename(trial_dir)
        # Join to make a data ID
        data_id = os.path.join(key_id, val_id)

        # Load results from knn, pre-edit metrics, and post-edit metrics
        restore_dir = os.path.join(trial_dir, 'models')
        knn_analysis_results = torch.load(os.path.join(restore_dir, 'knn_analysis_results.pth'))
        pre_edit_metrics = torch.load(os.path.join(restore_dir, 'pre_edit_metrics.pth'))
        post_edit_metrics = torch.load(os.path.join(restore_dir, 'post_edit_metrics.pth'))

        # Combine results into one dictionary
        combined_results = combine_results(
            data_id=data_id,
            knn_analysis=knn_analysis_results,
            pre_edit_metrics=pre_edit_metrics,
            post_edit_metrics=post_edit_metrics)

        # Save column headers in first trial run
        if trial_idx == 0:
            column_headers = list(combined_results.keys())
        # Convert results to np.array & append to list
        combined_results = np.expand_dims(np.array(list(combined_results.values())), axis=0)
        data.append(combined_results)

    # Convert data from list of np.arrays -> pd.DataFrame
    data = np.concatenate(data, axis=0)
    df = pd.DataFrame(data, columns=column_headers)

    df.to_csv(save_path)
    print("Saved CSV to {}".format(save_path))

    return df

if __name__ == "__main__":
    parser.add_argument('--save_dir',
        type=str, required=True, help='Directory to find trial_paths.txt')
    parser.add_argument('--class_list_path',
        type=str, default='metadata/cinic-10/class_names.txt', help='Path to text file with list of class names in order')
    # save_dir = 'saved/edit/trials/CINIC10_ImageNet-VGG_16/0112_163516'

    args = parser.parse_args()
    trial_paths_path = os.path.join(args.save_dir, 'trial_paths.txt')
    # class_list_path = 'metadata/cinic-10/class_names.txt'

    trial_paths = read_lists(trial_paths_path)
    class_list = read_lists(args.class_list_path)
    save_path = os.path.join(args.save_dir, 'results_table.csv')

    print("Reading paths from {}".format(trial_paths_path))

    store_csv(
        trial_dirs=trial_paths,
        class_list=class_list,
        save_path=save_path)
