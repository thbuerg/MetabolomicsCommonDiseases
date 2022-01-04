import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, LogNormal, Weibull, transform_to

import numpy as np


def cox_ph_loss(logh, durations, events, eps=1e-7):
    """
    Simple approximation of the COX-ph. Log hazard is not computed on risk-sets, but on ranked list instead.
    This approximation is valid for datamodules w/ low percentage of ties.
    Credit to Haavard Kamme/PyCox
    :param logh:
    :param durations:
    :param events:
    :param eps:
    :return:
    """
    # sort:
    idx = durations.sort(descending=True, dim=0)[1]
    events = events[idx].squeeze(-1)
    logh = logh[idx].squeeze(-1)
    # calculate loss:
    gamma = logh.max()
    log_cumsum_h = logh.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    if events.sum() > 0:
        loss = - logh.sub(log_cumsum_h).mul(events).sum().div(events.sum())
    else:
        loss = - logh.sub(log_cumsum_h).mul(events).sum()
    return loss

def DSM_uncensored_loss(logf_ts, ks, events, e=1):
    """
    We minimize the ELBO of log P(DATASET_uncensored)
    equalling the negative sum over all log hazards.
    inputs are expected to be 2D Tensors of shape [B, k_dim]
    :param logf_t:
    :param durations:
    :param events:
    :return:
    """

    e_ = torch.Tensor([e])
    e_ = e_.type_as(logf_ts)
    zero_ = torch.Tensor([0])
    zero_ = zero_.type_as(logf_ts)

    elbo = torch.logsumexp(F.log_softmax(ks, dim=1)+logf_ts, dim=1, keepdim=True)
    mask = torch.eq(events, e_)
    elbo = elbo[mask]

    if torch.eq(mask.sum(), zero_):
        return torch.Tensor([1.0]).squeeze().type_as(logf_ts)
    else:
        return -elbo.sum() / (mask.sum())


def DSM_censored_loss(logS_ts, ks, events, e=1):
    """
    NLL on log hazards.

    For competing risks, all other events are treated as administrative censoring.

    :param logh:
    :param durations:
    :param events:
    :return:
    """
    e_ = torch.Tensor([e])
    e_ = e_.type_as(logS_ts)

    elbo = torch.logsumexp(F.log_softmax(ks, dim=1)+logS_ts, dim=1, keepdim=True)
    mask = torch.ne(events, e_)
    elbo = elbo[mask]

    return -elbo.sum()/mask.sum()
