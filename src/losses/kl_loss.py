from __future__ import print_function

import torch
import torch.nn as nn

"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""

def kl_loss(features, softlabels, hard_labels, temperature=0.07, base_temperature=0.07, device = None):
    """Compute loss for model. 
    Args:
        features: hidden vector of shape [bsz, hide_dim].
        soft_labels : hidden vector of shape [bsz, hide_dim].
        labels: ground truth of shape [bsz].
    Returns:
        A loss scalar.
    """
    if device is not None:
        device = device
    else :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # compute logits
    # print(hard_labels.shape)
    features_dot_softlabels = torch.div(torch.matmul(features, softlabels.T), temperature) # b 300
    features_dot_hardlabels = torch.div(torch.bmm(features.unsqueeze(1), hard_labels.permute(0, 2, 1 )), temperature).squeeze()  # b 300
    loss = torch.nn.functional.kl_div(features_dot_softlabels.softmax(-1).log(), features_dot_hardlabels.softmax(-1))


    return loss


def kl_loss2(features, softlabels, hard_labels, temperature=0.07, base_temperature=0.07, device=None):
    """Compute loss for model.
    Args:
        features: hidden vector of shape [bsz, hide_dim].
        soft_labels : hidden vector of shape [bsz, hide_dim].
        labels: ground truth of shape [bsz].
    Returns:
        A loss scalar.
    """
    if device is not None:
        device = device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # compute logits
    # print(hard_labels.shape)
    features_dot_softlabels = torch.div(torch.matmul(features, softlabels.T), temperature)  # b 300
    features_dot_hardlabels = torch.div(torch.matmul(features, hard_labels.T), temperature)  # b 300
    loss = torch.nn.functional.kl_div(features_dot_softlabels.softmax(-1).log(), features_dot_hardlabels.softmax(-1))

    return loss
