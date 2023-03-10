import numpy as np
import pandas as pd
import os, sys
import torch
from sklearn import metrics
from scipy import stats

sys.path.insert(0, 'src')

def get_IOU(predictions_a, 
            predictions_b, 
            target_class_idx=None, 
            modes=[]):
    '''
    Given predictions and list of IOU modes, return list of IOU values
    
    Arg(s):
        predictions_a : N-length np.array or torch.tensor
            First list of predictions
        predictions_b : N-length np.array or torch.tensor
            Second list of predictions
        target_class_idx : int or None
            index of target class, only needed for binary IOU
        modes : list[str]
            options: ['binary', 'weighted', 'macro', 'micro']
            
    Returns:
        IOUs : list[float]
            list of IOUs corresponding to modes list
    '''
    
    IOUs = []
    
    # Assert valid modes
    for mode in modes:
        assert mode in ['binary', 'weighted', 'macro', 'micro']
        
    if torch.is_tensor(predictions_a):
        predictions_a = predictions_a.cpu().numpy()
    if torch.is_tensor(predictions_b):
        predictions_b = predictions_b.cpu().numpy()
    
    # Calculate each IOU
    for mode in modes:
        try:
            if mode == 'binary':
                # Binarize predictions based on target class
                binary_predictions_a = np.where(
                    predictions_a == target_class_idx,
                    1, 0)
                binary_predictions_b = np.where(
                    predictions_b == target_class_idx,
                    1, 0)
                
                IOU = metrics.jaccard_score(
                    y_true=binary_predictions_a,
                    y_pred=binary_predictions_b,
                    average=mode)
            else:
                IOU = metrics.jaccard_score(
                    y_true=predictions_a,
                    y_pred=predictions_b,
                    average=mode)
            IOUs.append(IOU)
        except Exception as e:
            print(e)
            continue
    return IOUs
    


def get_spearman(logits_a, logits_b, target_class_idx):
    '''
    Given the logits from 2 models and desired class, return Spearman correlation of rankings
    
    Arg(s):
        logits_a : N x C np.array or torch.tensor
            first logit output for N samples and C classes
        logits_b : N x C np.array or torch.tensor
            second logit output for N samples and C classes
        target_class_idx : int
            index to calculate Spearman's for
    
    Returns:
        spearman : stats.spearman object
            spearman.correlation
            spearman.pvalue
    '''
    def get_ranking(logits, target_class_idx):
        if not torch.is_tensor(logits):
            logits = torch.from_numpy(logits)

        softmax = torch.softmax(logits, dim=1)
        target_softmax = softmax[:, target_class_idx]
        ranking = target_softmax.argsort().argsort()

        return ranking.cpu().numpy()
    ranking_a = get_ranking(
        logits=logits_a,
        target_class_idx=target_class_idx)
    ranking_b = get_ranking(
        logits=logits_b,
        target_class_idx=target_class_idx)
    
    if torch.is_tensor(ranking_a):
        ranking_a = ranking_a.cpu().numpy()
    if torch.is_tensor(ranking_b):
        ranking_b = ranking_b.cpu().numpy()
    spearman = stats.spearmanr(
        a=ranking_a,
        b=ranking_b)
    
    return spearman