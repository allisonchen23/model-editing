import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, 'src')
from utils.visualizations import histogram, bar_graph

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

def convert_string_columns(df_, columns=None):
    '''
    Given a dataframe, convert columns to numpy if they are strings

    Arg(s):
        df_ : pd.DataFrame
            Original dataframe
        columns : list[str] or None
            If None, iterate all the columns

    Returns:
        df_ : modified dataframe with strings replaced with numpy.array
    '''

    if columns == None:
        columns = df_.columns

    for column in columns:
        df_[column] = df_[column].map(string_to_numpy)
    return df_

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

def summary_histogram(df_,
                     metrics=None,
                     n_bins=10,
                     save_dir=None,
                     tag=None):
    '''
    Display/save histograms of distributions of the metrics provided
    '''

    if metrics is None:
        metrics = [['{} Accuracy', '{} Mean Precision', '{} Mean Recall', '{} Mean F1'],
                   ['{} Target Precision', '{} Target Recall', '{} Target F1'],
                   ['{} Orig Pred Precision', '{} Orig Pred Recall', '{} Orig Pred F1']]

    mean_df_ = df_.mean()
    for row in metrics:
        for metric in row:
            pre_metric_mean = mean_df_[metric.format("Pre")]
            post_metric = df_[metric.format("Post")].to_numpy()
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

def compare_histogram()