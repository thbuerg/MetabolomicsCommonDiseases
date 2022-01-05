import os
import json
import warnings

import hydra
import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
import neptune.new as neptune

from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.nn import Sigmoid, SELU, ReLU
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, MultiStepLR
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging

from metabolomicstatemodel.source.datamodules import *
from metabolomicstatemodel.source.tasks import *
from metabolomicstatemodel.source.modules import *
from metabolomicstatemodel.source.utils import set_up_neptune, get_default_callbacks
from metabolomicstatemodel.source.callbacks import WriteCheckpointLogs, WritePredictionsDataFrame, LogCoxBaseline, WriteLatentsDataFrame


# globals:
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.use_inf_as_na = True
pl.seed_everything(23)  #the number of doom


assert os.environ['NEPTUNE_API_TOKEN'], 'No Neptune API Token found. Please do `export NEPTUNE_API_TOKEN=<token>`.'
config_path = "config/"


@hydra.main(config_path, config_name="config")
def train(FLAGS: DictConfig):
    OmegaConf.set_struct(FLAGS, False)
    FLAGS.config_path = config_path

    # get classes
    Task = eval(FLAGS.experiment.task)
    Module = eval(FLAGS.experiment.module)
    DataModule = eval(FLAGS.experiment.datamodule)
    if FLAGS.experiment.latent_module is not None:
        LatentModule = eval(FLAGS.experiment.latent_module)
    else:
        LatentModule = None

    # initialize datamodule
    # load features.yaml if necessary:
    if FLAGS.experiment.feature_set is not None:
        FLAGS.experiment.features = OmegaConf.load(os.path.join(FLAGS.config_path, FLAGS.experiment.features_yaml))
    datamodule = DataModule(**FLAGS.experiment)
    datamodule.prepare_data()
    datamodule.setup("fit")
    FLAGS["data"] = {"feature_names": datamodule.features}

    # get network:
    ft_extractor = Module(input_dim=len(datamodule.features), **FLAGS.experiment.module_kwargs)
    if LatentModule is not None:
        if LatentModule == ResidualHeadMLP:
            FLAGS.experiment.latent_module_kwargs.skip_connection_input_dim = len(datamodule.features)
        cause_specific = LatentModule(**FLAGS.experiment.latent_module_kwargs)
    else:
        cause_specific = nn.Identity()

    # initialize Task
    task = Task(feature_extractor=ft_extractor,
                latent_mlp=cause_specific,
                feature_dim=len(datamodule.features),
                **FLAGS.experiment)

    # initialize trainer
    callbacks = get_default_callbacks(monitor=FLAGS.experiment.monitor) \
    callbacks += [WriteCheckpointLogs(),
                  WritePredictionsDataFrame(write_calibrated_predictions=FLAGS.experiment.write_calibrated_predictions)]

    trainer = pl.Trainer(**FLAGS.trainer,
                         callbacks=callbacks,
                         logger=set_up_neptune(FLAGS))

    FLAGS["parameters/callbacks"] = [c.__class__.__name__ for c in callbacks]
    trainer.logger.run["FLAGS"] = FLAGS

    if FLAGS.trainer.auto_lr_find:
        trainer.tune(model=task, datamodule=datamodule)

    # run
    trainer.fit(task, datamodule)
    trainer.logger.run.stop()


if __name__ == '__main__':
    main()
