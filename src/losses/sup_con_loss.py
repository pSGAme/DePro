from __future__ import print_function

import numpy as np
import torch
torch.set_printoptions(threshold=np.inf)
import torch.nn as nn

import torch.nn.functional as F

"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""


def sup_con_loss(features, temperature=0.07, contrast_mode='all', base_temperature=0.07, labels=None, mask=None,
                 device=None):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    if device is not None:
        device = device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)  # 对角线全1的矩阵
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]  # 4 batch = 10
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # 40,50
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature)  # 40 40
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 40,1
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)  # 40, 40
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )  # 对角线为0，其余为1
    mask = mask * logits_mask  # 把对角线的1全变为0

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask  # 除了对角线都有值
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # 每一行表示ancher和除自己以外的feature之和,然后logit中减去这个值

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # 40

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()  # 4,10 -> mean

    return loss

def domain_specific_sup_con_loss(features, temperature=0.07, contrast_mode='all', base_temperature=0.07, labels=None, mask=None,
                 device=None):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    if device is not None:
        device = device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)  # 对角线全1的矩阵
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]  # 4 batch = 10
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # 40,50
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature)  # 40 40
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 40,1
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)  # 40, 40
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )  # 对角线为0，其余为1
    mask = mask * logits_mask  # 把对角线的1全变为0

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask  # 除了对角线都有值
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # 每一行表示ancher和除自己以外的feature之和,然后logit中减去这个值

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # 40

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()  # 4,10 -> mean

    return loss

# if self.distance == 'euclidean':
#             dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
#             dist = dist + dist.t()
#             dist.addmm_(1, -2, inputs, inputs.t())
#             dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
#         elif self.distance == 'consine':
#             fnorm = torch.norm(inputs, p=2, dim=1, keepdim=True)
#             l2norm = inputs.div(fnorm.expand_as(inputs))
#             dist = - torch.mm(l2norm, l2norm.t())
#
#         if self.use_gpu: targets = targets.cuda()
#         # For each anchor, find the hardest positive and negative
#         mask = targets.expand(n, n).eq(targets.expand(n, n).t())
#         dist_ap, dist_an = [], []
#         for i in range(n):
#             dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
#             dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
#         dist_ap = torch.cat(dist_ap)
#         dist_an = torch.cat(dist_an)
#
#         # Compute ranking hinge loss
#         y = torch.ones_like(dist_an)
#         loss = self.ranking_loss(dist_an, dist_ap, y)
#         return loss


def triplet_loss_unnorm(features, target, temperature=1, margin=0.5):
    # score = torch.div(torch.matmul(features, features.T), temperature) # b 300

    inputs = features
    n = inputs.shape[0]
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs, inputs.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    target1 = target.unsqueeze(1)
    mask = target1 == target1.t()
    pair_labels = mask.float()

    max_pos = torch.max(dist * pair_labels -
                        (1 - pair_labels + torch.eye(dist.size(0), device=dist.device)) * 1e5, dim=1)[0]

    min_neg = torch.min(dist * (1 - pair_labels) + pair_labels * 1e5, dim=1)[0]
    # Compute ranking hinge loss
    loss = F.margin_ranking_loss(min_neg, max_pos, torch.ones_like(target), margin=margin)

    return loss, loss


def triplet_loss(features, target, temperature=1, margin=0.5):
    score = torch.div(torch.matmul(features, features.T), temperature)  # b 300

    target1 = target.unsqueeze(1)
    mask = target1 == target1.t()
    pair_labels = mask.float()

    min_pos = torch.min(score * pair_labels +  # get the positive scores
                        (1 - pair_labels + torch.eye(score.size(0), device=score.device)) * 1e15, dim=1)[0]

    max_neg = torch.max(score * (1 - pair_labels) - pair_labels * 1e15, dim=1)[0]
    # Compute ranking hinge loss
    loss = F.margin_ranking_loss(min_pos, max_neg, torch.ones_like(target), margin=margin)

    with torch.no_grad():
        min_pos = torch.min(score * pair_labels +
                            (1 - pair_labels + torch.eye(score.size(0), device=score.device)) * 1e15, dim=1)[0]
        max_neg = torch.max(score * (1 - pair_labels) - pair_labels * 1e15, dim=1)[0]
        acc = (min_pos > max_neg).float()
    return loss, acc


def domain_aware_triplet_loss_v1(features, target, domain_ids, temperature=1, margin=0.5):
    score = torch.div(torch.matmul(features, features.T), temperature)  # b b
    target1 = target.unsqueeze(1)
    mask = target1 == target1.t()
    pair_labels = mask.float()

    domain_mask = domain_ids == domain_ids.t()
    domain_mask = domain_ids.float()

    min_pos = torch.min(score * pair_labels +  # get the positive scores
                        (1 - pair_labels + torch.eye(score.size(0), device=score.device)) * 1e15, dim=1)[0]
    min_pos_index = torch.min(score * pair_labels +  # get the positive scores
                              (1 - pair_labels + torch.eye(score.size(0), device=score.device)) * 1e15, dim=1)[1]
    min_pos_domain = domain_ids[min_pos_index]

    min_pos_rate_diff = (min_pos_domain != domain_ids).sum() / domain_ids.shape[
        0]  # positive, from other domain is better

    max_neg = torch.max(score * (1 - pair_labels) - pair_labels * 1e15, dim=1)[0]
    max_neg_index = torch.max(score * (1 - pair_labels) - pair_labels * 1e15, dim=1)[1]
    max_neg_domain = domain_ids[max_neg_index]
    max_neg_rate_same = (max_neg_domain == domain_ids).sum() / domain_ids.shape[0]

    # Compute ranking hinge loss
    loss = F.margin_ranking_loss(min_pos, max_neg, torch.ones_like(target), margin=margin)

    return loss, min_pos_rate_diff, max_neg_rate_same


def domain_aware_triplet_loss_v2(features, target, domain_ids, temperature=1, margin=0.5):
    score = torch.div(torch.matmul(features, features.T), temperature)  # b b
    target1 = target.unsqueeze(1)
    mask = target1 == target1.t()
   # print(target, domain_ids)
    pair_labels = mask.float()

    domain_ids = domain_ids.unsqueeze(1)
    domain_mask = domain_ids == domain_ids.t()
    domain_mask = domain_mask.float()
    # print(domain_mask)
    # print(pair_labels)
    min_pos = torch.min(score * (1 - domain_mask) * pair_labels +  # get the positive scores
                        (1 - (1 - domain_mask) * pair_labels) * 1e15,
                        dim=1)[0]
    min_pos_index = torch.min(score * (1 - domain_mask) * pair_labels +  # get the positive scores
                              (1 - (1 - domain_mask) * pair_labels) * 1e15,
                              dim=1)[1]
    min_pos_domain = domain_ids[min_pos_index]
    min_pos_rate_diff = (min_pos_domain != domain_ids).sum() / domain_ids.shape[0]  # positive, from other domain is
    # better

    # max_neg = torch.max(score * (1 - domain_mask) * (1 - pair_labels) - (1 - (1 - domain_mask) * (1 - pair_labels)) * 1e15, dim=1)[0]
    # max_neg_index = torch.max(score * (1 - domain_mask) * (1 - pair_labels) - (1 - (1 - domain_mask) * (1 - pair_labels)) * 1e15, dim=1)[1]
    # max_neg_domain = domain_ids[max_neg_index]
    # max_neg_rate_same = (max_neg_domain == domain_ids).sum() / domain_ids.shape[0]
    max_neg = torch.max(score * (1 - pair_labels) - pair_labels * 1e15, dim=1)[0]
    max_neg_index = torch.max(score * (1 - pair_labels) - pair_labels * 1e15, dim=1)[1]
    max_neg_domain = domain_ids[max_neg_index]
    max_neg_rate_same = (max_neg_domain == domain_ids).sum() / domain_ids.shape[0]
    # Compute ranking hinge loss
    loss = F.margin_ranking_loss(min_pos, max_neg, torch.ones_like(target), margin=margin)

    return loss, min_pos_rate_diff, max_neg_rate_same


def pairwise_matching_loss(features, labels, temperature=1):
    score = torch.div(torch.matmul(features, features.T), temperature)  # b 300
    labels = labels.unsqueeze(1)
    mask = (labels == labels.t())
    pair_labels = mask.float()
    loss = F.binary_cross_entropy_with_logits(score, pair_labels)
    # loss = loss.sum(-1)

    with torch.no_grad():
        min_pos = torch.min(score * pair_labels +
                            (1 - pair_labels + torch.eye(score.size(0), device=score.device)) * 1e15, dim=1)[0]
        max_neg = torch.max(score * (1 - pair_labels) - pair_labels * 1e15, dim=1)[0]
        acc = (min_pos > max_neg).float()

    return loss, acc


def soft_sup_con_loss(features, softlabels, hard_labels, temperature=0.07, base_temperature=0.07, device=None):
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

    # hard_labels = hard_labels.contiguous().view(-1, 1)
    # mask = torch.eq(hard_labels, hard_labels.T).float().to(device) # batch, batch

    # compute logits
    features_dot_softlabels = torch.div(torch.matmul(features, softlabels.T), temperature)  # b 300
    predict = torch.argmax(features_dot_softlabels, 1)
    correct = (predict == hard_labels).sum().item()
    loss = torch.nn.functional.cross_entropy(features_dot_softlabels, hard_labels)

    return loss, correct


def domain_specific_soft_sup_con_loss(features, softlabels, hard_labels, temperature=0.07, base_temperature=0.07, device=None):
    """Compute loss for model.
    Args:
        features: hidden vector of shape [bsz, hide_dim].
        soft_labels : hidden vector of shape [300, hide_dim].
        labels: ground truth of shape [bsz].
    Returns:
        A loss scalar.
    """
    if device is not None:
        device = device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hard_labels = hard_labels.contiguous().view(-1, 1)
    # mask = torch.eq(hard_labels, hard_labels.T).float().to(device) # batch, batch

    # compute logits
    #print(len(softlabels))
    features_dot_softlabels = []
    for i in range(5):
        small_bs = features.shape[0] // 5
        softlabel = softlabels[i]
        feature = features[i*small_bs: (i+1)*small_bs]
       # print(softlabel.shape, feature.shape)
        feature_dot_softlabel = torch.div(torch.matmul(feature, softlabel.T), temperature)  # b 300
        #print(feature_dot_softlabel.shape)
        features_dot_softlabels.append(feature_dot_softlabel)
    features_dot_softlabels = torch.cat(features_dot_softlabels, dim=0)
    #print(features_dot_softlabels.shape)
    # features_dot_softlabels = torch.div(torch.matmul(features, softlabels.T), temperature)  # b 300

    predict = torch.argmax(features_dot_softlabels, 1)
    correct = (predict == hard_labels).sum().item()
    loss = torch.nn.functional.cross_entropy(features_dot_softlabels, hard_labels)

    return loss, correct
