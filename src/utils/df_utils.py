import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, 'src')
from utils.visualizations import histogram, bar_graph

def load_and_preprocess_csv(csv_path,
             drop_duplicates=None,
             round_to=None,
             string_array_columns=None):
    '''
    Given path to csv and optional pre-processing, return corresponding df.

    Arg(s):
        csv_path : str
            path to CSV file
        drop_duplicates : None or list[str]
            if not None, columns by which to drop duplicates based off of
        round_to : None or int
            number of decimal places to round floats to
        string_array_columns : None or list[str]
            if not None, columns to turn from string array -> float array

    Returns:
        df : pd.DataFrame
            processed data frame
    '''

    df = pd.read_csv(csv_path)
    if drop_duplicates is not None:
        assert type(drop_duplicates) == list, "Invalid type {} for drop_duplicates. Expected list".format(type(drop_duplicates))
        df = df.drop_duplicates(subset=drop_duplicates)
        # Sanity checks
        set_rows = set(df[drop_duplicates])
        list_rows = list(df[drop_duplicates])
        assert len(set_rows) == len(list_rows)

    if round_to is not None:
        assert type(round_to) == int, "Invalid type {} for round_to. Expected list".format(type(round_to))
        df = df.round(round_to)

    df = convert_string_columns(df, columns=string_array_columns)

    return df

def string_to_numpy(string, verbose=False):
    '''
    Given a string, convert it to a numpy array

    Arg(s):
        string : str
            string assumed in format of numbers separated by spaces, with '[' ']' on each end
        verbose : bool
            whether or not to print error messages
    Returns:
        np.array
    '''
    if type(string) != str:
        return string
    original_string = string

    if string[0] == '[':
        string = string[1:]
    if string[-1] == ']':
        string = string[:-1]

    string = string.split()
    try:
        string = [eval(i) for i in string]
    except:
        if verbose:
            print("Unable to convert '{}' to numbers. Returning original string".format(original_string))
        return original_string

    return np.array(string)

def convert_string_columns(df, columns=None):
    '''
    Given a dataframe, convert columns to numpy if they are strings

    Arg(s):
        df : pd.DataFrame
            Original dataframe
        columns : list[str] or None
            If None, iterate all the columns

    Returns:
        df : modified dataframe with strings replaced with numpy.array
    '''

    if columns == None:
        columns = df.columns

    for column in columns:
        df[column] = df[column].map(string_to_numpy)
    return df

def mean_numpy_series(series, axis=0):
    '''
    Given a series of numpy arrays, return the mean

    Arg(s):
        series : pd.Series
            series to take mean of
        axis : int
            axis across which to take the mean. Default is 0

    '''
    data = np.array(list(series))
    return np.mean(data, axis=axis)

def summary_histogram(df,
                     metrics=None,
                     n_bins=10,
                     save_dir=None,
                     tag=None):
    '''
    Display/save histograms of distributions of the metrics provided

    Arg(s):
        df : pd.DataFrame
            Data fram as result of results_to_csv.py
        metrics : list[list[str]]
            list of list of metrics with Pre/Post replaced by {}
        n_bins : int
            number of bins in histogram
        save_dir : str or None
            directory to save histogram to
        tag : str or None
            label in front of metric name when saving histogram.

    '''

    if metrics is None:
        metrics = [['{} Mean Accuracy', '{} Mean Precision', '{} Mean Recall', '{} Mean F1'],
                   ['{} Target Accuracy', '{} Target Precision', '{} Target Recall', '{} Target F1'],
                   ['{} Orig Pred Accuracy', '{} Orig Pred Precision', '{} Orig Pred Recall', '{} Orig Pred F1']]

    mean_df = df.mean()
    for row in metrics:
        for metric in row:
            try:
                pre_metric_mean = mean_df[metric.format("Pre")]
                post_metric = df[metric.format("Post")].to_numpy()
            except:
                print("Unable to find metric '{}' in dataframe".format(metric))
                continue
            # Create save directory
            if save_dir is not None:
                if tag is None:
                    save_path = os.path.join(save_dir, metric.format("").strip())
                else:
                    save_path = os.path.join(save_dir, tag + "_" + metric.format("").strip())
            else:
                save_path = None

            histogram(
                data=post_metric,
                n_bins=n_bins,
                marker=pre_metric_mean,
                title=metric.format("Post"),
                xlabel=metric.split(" ", maxsplit=1)[1],
                ylabel="Counts",
                save_path=save_path)

def get_sorted_idxs(df,
                    columns,
                    id_column='Unnamed: 0',
                    increasing=True):
    '''
    Given a data frame and a column to sort by, sort dataframe and return sorted indices

    Arg(s):
        df : pd.DataFrame
            data frame
        columns : str or list[str]
            columns to sort by
        id_column : str
            the column name of ID numbers
        increasing : bool
            if True, sort in increasing order. Else, decreasing

    Returns:
        sorted_df, sorted_idxs : pd.DataFrame, np.array
            sorted dataframe
            list of indices in sorted order
    '''

    sorted_df = df.sort_values(
        columns,
        ascending=increasing)
    sorted_idxs = sorted_df[id_column].to_numpy()

    return sorted_df, sorted_idxs