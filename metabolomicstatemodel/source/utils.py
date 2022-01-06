import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from .logging import FoolProofNeptuneLogger


####################################################################################################
#                        neptune                                                                   #
####################################################################################################

def set_up_neptune(FLAGS={}, close_after_fit=False, **kwargs):
    """
    Set up a neptune logger from file.
    :param keyfile:
    :param project:
    :param name:
    :param params:
    :param tags:
    :param close_after_fit:
    :param kwargs:
    :return:
    """
    if not "NEPTUNE_API_TOKEN" in os.environ:
        raise EnvironmentError('Please set environment variable `NEPTUNE_API_TOKEN`.')

    neptune_logger = FoolProofNeptuneLogger(api_key=os.environ["NEPTUNE_API_TOKEN"],
                                            close_after_fit=close_after_fit,
                                            **FLAGS.setup)
    return neptune_logger


def get_default_callbacks(monitor='Ctd_0.9', mode='max', early_stop=True):
    """
    Instantate the default callbacks: EarlyStopping and Checkpointing.

    :param monitor:
    :param mode:
    :return:
    """
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor=monitor, verbose=True,
                                                                        save_last=True, save_top_k=3,
                                                                        save_weights_only=False, mode=mode,
                                                                        period=1)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=False)
    if early_stop:
        early_stop = pl.callbacks.early_stopping.EarlyStopping(monitor=monitor, min_delta=1e-5, patience=15,
                                                               verbose=True, mode=mode, strict=False)
        return [checkpoint_callback, early_stop, lr_monitor]
    else:
        return [checkpoint_callback, lr_monitor]


def attribution2df(attributions, feature_names, loader):
    attribution_sum = attributions.detach().numpy().sum(0)
    attribution_norm_sum = attribution_sum / np.linalg.norm(attribution_sum, ord=1)
    axis_data = np.arange(loader.shape[1])
    data_labels = list(map(lambda idx: feature_names[idx], axis_data))
    df = pd.DataFrame({'feature': data_labels,
                       'importance': attribution_norm_sum})
    sorted_df = df.reindex(df.importance.abs().sort_values(ascending=False).index)
    return sorted_df

