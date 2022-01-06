import random

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, MultiStepLR
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data
from tqdm.auto import tqdm
from copy import deepcopy
from omegaconf.listconfig import ListConfig

from sksurv.metrics import concordance_index_ipcw
from lifelines.utils import concordance_index
from lifelines.exceptions import ConvergenceError
from sklearn.isotonic import IsotonicRegression

from .losses import *
from .evaluation import get_observed_probability
from .datasets import BatchedDS


class AbstractSurvivalTask(pl.LightningModule):
    """
    Defines a Task (in the sense of Lightning) to train a CoxPH-Model.
    """

    def __init__(self, network,
                 batch_size=128,
                 num_workers=8,
                 lr=1e-3,
                 evaluation_time_points=[5, 10],
                 evaluation_quantile_bins=None,
                 report_train_metrics=True,
                 optimizer=torch.optim.SGD,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={}, **kwargs):
        """
        Defines a Task (in the sense of Lightning) to train a CoxPH-Model.

        :param network: `nn.Module or pl.LightningModule`,  the network that should be used.
        :param batch_size:  `int`, batchsize
        :param num_workers: `int`, num_workers for the DataLoaders
        :param optimizer:   `torch.optim`, class, is instantiated w/ the passed optimizer args by trainer.
        :param optimizer_kwargs:    `dict`, optimizer args.
        :param schedule:    `scheudle calss` to use
        :param schedule_kwargs:  `dict`, schedule kwargs, like: {'patience':10, 'threshold':0.0001, 'min_lr':1e-6}
        """
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.net = network

        self.lr = lr
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        self.schedule = schedule
        self.schedule_kwargs = schedule_kwargs
        self.evaluation_quantile_bins = eval(evaluation_quantile_bins) if not isinstance(evaluation_quantile_bins, (type(None), list, ListConfig)) else evaluation_quantile_bins
        self.evaluation_time_points = eval(evaluation_time_points) if not isinstance(evaluation_time_points, (type(None), list, ListConfig)) else evaluation_time_points

        self.params = []
        self.networks = [self.net]
        for n in self.networks:
            if n is not None:
                self.params.extend(list(n.parameters()))

        # save the params.
        self.save_hyperparameters()

    def unpack_batch(self, batch):
        data, (durations, events) = batch
        return data, durations, events

    def configure_optimizers(self):
        if isinstance(self.optimizer, str): self.optimizer = eval(self.optimizer)
        if isinstance(self.schedule, str): self.schedule = eval(self.schedule)
        self.optimizer_kwargs["lr"] = self.lr

        optimizer = self.optimizer(self.params, **self.optimizer_kwargs)
        print(f'Using Optimizer {str(optimizer)}.')
        if self.schedule is not None:
            print(f'Using Scheduler {str(self.schedule)}.')
            schedule = self.schedule(optimizer, **self.schedule_kwargs)
            if isinstance(self.schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    'optimizer': optimizer,
                    'scheduler': schedule,
                    'monitor': 'Ctd_0.9',
                }
            else:
                return [optimizer], [schedule]
        else:
            print('No Scheduler specified.')
            return optimizer

    def ext_dataloader(self, ds, batch_size=None, shuffle=False, num_workers=None,
                       drop_last=False):  ### Already transformed datamodules? -> Or pass transformers?
        if batch_size is None:
            batch_size = self.batch_size
        if num_workers is None:
            num_workers = self.num_workers
        return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                          shuffle=shuffle, drop_last=drop_last)

    def training_step(self, batch, batch_idx):
        data, durations, events = self.unpack_batch(batch)
        args = self.shared_step(data, durations, events)
        loss_dict = self.loss(*args)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        data, durations, events = self.unpack_batch(batch)
        args = self.shared_step(data, durations, events)
        loss_dict = self.loss(*args)
        loss_dict = dict([(f'val_{k}', v) for k, v in loss_dict.items()])
        for k, v in loss_dict.items():
            self.log(k, v, on_step=False, on_epoch=False, prog_bar=False, logger=False)
        return loss_dict

    # def on_train_epoch_end(self, outputs):
    #     raise NotImplementedError('train epoch end')
    # return loss_dict['loss']

    def validation_epoch_end(self, outputs):
        metrics = {}
        # aggregate the per-batch-metrics:
        for metric_name in ["val_loss"]:
            # for metric_name in [k for k in outputs[0].keys() if k.startswith("val")]:
            metrics[metric_name] = torch.stack([output[metric_name] for output in outputs]).mean()

        # calculate the survival metrics
        valid_ds = self.val_dataloader().dataset if not \
            isinstance(self.val_dataloader().dataset, BatchedDS) \
            else self.val_dataloader().dataset.dataset
        train_ds = self.train_dataloader().dataset if not \
            isinstance(self.train_dataloader().dataset, BatchedDS) \
            else self.train_dataloader().dataset.dataset
        metrics_survival = self.calculate_survival_metrics(train_ds=train_ds, valid_ds=valid_ds,
                                                           time_points=self.evaluation_time_points,
                                                           quantile_bins = self.evaluation_quantile_bins)
        # train metrics:
        if self.hparams.report_train_metrics:
            train_metrics_survival = self.calculate_survival_metrics(train_ds=train_ds, valid_ds=train_ds,
                                                                     time_points = self.evaluation_time_points,
                                                                     quantile_bins = self.evaluation_quantile_bins)
            for key, value in train_metrics_survival.items():
                metrics[f'train_{key}'] = value

        metrics.update(metrics_survival)

        for key, value in metrics.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def calculate_survival_metrics(self, train_ds, valid_ds, time_points=[5, 10], quantile_bins=None):
        """
        Calculate epoch level survival metrics.
        :param train_ds:
        :param valid_ds:
        :param time_points: times at which to evaluate.
        :param quantile_bins: ALTERNATIVELY (to time_points) -> pass quantiles of the time axis.
        :return:
        """
        metrics = {}
        ctds = []
        cs = []

        assert None in [time_points, quantile_bins], 'EITHER pass quantiles OR pass timepoints'

        try:
            surv_train = np.stack([train_ds.events.values, train_ds.durations.values], axis=0).squeeze(axis=-1)
            surv_valid = np.stack([valid_ds.events.values, valid_ds.durations.values], axis=0).squeeze(axis=-1)
        except AttributeError:
            surv_train = np.stack([train_ds.events, train_ds.durations], axis=0).squeeze(axis=-1)
            surv_valid = np.stack([valid_ds.events, valid_ds.durations], axis=0).squeeze(axis=-1)

        # move to structured arrays:
        struc_surv_train = np.array([(e, d) for e, d in zip(surv_train[0], surv_train[1])],
                                    dtype=[('event', 'bool'), ('duration', 'f8')])
        struc_surv_valid = np.array([(e, d) for e, d in zip(surv_valid[0], surv_valid[1])],
                                    dtype=[('event', 'bool'), ('duration', 'f8')])

        self.eval()
        loader = self.ext_dataloader(valid_ds, batch_size=32, num_workers=0, shuffle=False, drop_last=False)

        if time_points is None:
            assert quantile_bins is not None, 'If timepoints is None, then pass quantile bins'
            taus = [np.quantile(surv_valid[1, surv_valid[0] > 0], q) for q in quantile_bins]
            annot = quantile_bins
        else:
            taus = time_points
            annot = time_points

        for i, tau in enumerate(taus):
            risks = []
            tau_ = torch.Tensor([tau])
            with torch.no_grad():
                for batch in loader:
                    data, durations, events = self.unpack_batch(batch)
                    risk = self.predict_risk(data, t=tau_)  # returns RISK (e.g. F(t))
                    risks.append(risk.detach().cpu().numpy())
            try:
                risks = np.concatenate(risks, axis=0)
            except ValueError:
                risks = np.asarray(risks)
            risks = risks.ravel()
            risks[pd.isna(risks)] = np.nanmax(risks)
            Ctd = concordance_index_ipcw(struc_surv_train, struc_surv_valid,
                                       risks,
                                       tau=tau, tied_tol=1e-8)
            C = concordance_index(event_times=surv_valid[1],
                                  predicted_scores=-risks,
                                  event_observed=surv_valid[0])
            ctds.append(Ctd[0])
            cs.append(C)

        self.train()

        for k, v in zip(annot, ctds):
            metrics[f'Ctd_{k}'] = v
        for k, v in zip(annot, cs):
            metrics[f'C_{k}'] = v

        return metrics

    def shared_step(self, data, duration, events):
        """
        shared step between training and validation. should return a tuple that fits in loss.
        :param data:
        :param durations:
        :param events:
        :return:
        """
        raise NotImplementedError("Abstract method")
        return durations, events, some_args

    def loss(self, predictions, durations, events):
        """
        Calculate Loss.
        :param predictions:
        :param durations:
        :param events:
        :return:
        """
        raise NotImplementedError("Abstract Class")
        loss1 = None
        loss2 = None
        loss = None
        return {'loss': loss,
                'loss1': loss1,
                'loss2': loss2,}

    def predict_dataset(self, ds, times):
        """
        Predict the survival function for each sample in the dataset at all durations in the dataset.

        Returns a pandas DataFrame where the rows are timepoints and the columns are the samples. Values are S(t|X)
        :param ds:
        :param times: a np.Array holding the times for which to calculate the risk.
        :return:
        """
        raise NotImplementedError("Abstract method")

    def fit_isotonic_regressor(self, ds, times, n_samples):
        if len(ds) < n_samples: n_samples = len(ds)
        sample_idx = random.sample([i for i in range(len(ds))], n_samples)
        sample_ds = torch.utils.data.Subset(ds, sample_idx)
        pred_df = self.predict_dataset(sample_ds, np.array(times)).dropna()
        for t in times:
            if hasattr(self, 'isoreg'):
                if f"1_{t}_Ft" in self.isoreg:
                    pass
                else:
                    for i, array in enumerate([pred_df[f"1_{t}_Ft"].values, pred_df["events"].values, pred_df["durations"].values]):
                        if len(list(np.argwhere(np.isnan(array))))>0:
                            print(i)
                            print(np.argwhere(np.isnan(array)))
                    F_t_obs, nan = get_observed_probability(pred_df[f"1_{t}_Ft"].values, pred_df["events"].values, pred_df["durations"].values, t)
                    self.isoreg[f"1_{t}_Ft"] = IsotonicRegression().fit(pred_df.drop(pred_df.index[nan])[f"1_{t}_Ft"].values, F_t_obs)
            else:
                F_t_obs, nan = get_observed_probability(pred_df[f"1_{t}_Ft"].values, pred_df["events"].values, pred_df["durations"].values, t)
                self.isoreg = {f"1_{t}_Ft": IsotonicRegression().fit(pred_df.drop(pred_df.index[nan])[f"1_{t}_Ft"].values, F_t_obs)}

    def predict_dataset_calibrated(self, ds, times):
        pred_df = self.predict_dataset(ds, np.array(times))
        for t in times:
            pred_df[f"1_{t}_Ft__calibrated"] = self.isoreg[f"1_{t}_Ft"].predict(pred_df[f"1_{t}_Ft"])
        return pred_df

    @auto_move_data
    def forward(self, X, t=None):
        """
        Predict a sample
        :return: f_t, F_t, S_t
        """
        raise NotImplementedError("Abstract method")

    @auto_move_data
    def predict_risk(self, X, t=None):
        """
        Predict risk for X. Risk and nothing else.

        :param X:
        :param t:
        :return:
        """
        raise NotImplementedError("Abstract method")


class MultiTaskSurvivalTraining(pl.LightningModule):
    """
    Abstract class for MultiTaskSurvivalTraining:

    covariates, durations, events = sample  # here durations and events are multidimensional
    feature_extractor(covariates) -> latent

    preds = []
    for t in tasks:
        head_t(covariates) -> survival_pred_t
        preds.append(survival_pred_t)

    loss(survival_pred1,2,3 ; targets1,2,3)  # Mutlidimensional _NOT_ necessarily competing!
    """
    def __init__(self,
                 feature_extractor=None,
                 batch_size=128,
                 evaluation_time_points=[5, 10],
                 evaluation_quantile_bins=None,
                 task_specific_exclusions=False,
                 num_workers=8,
                 num_testtime_views=1,
                 latent_dim=100,
                 latent_mlp=None,
                 feature_dim=10,
                 task_names=None,
                 task_weights=None,
                 n_tasks=None,
                 survival_task=None,
                 survival_task_kwargs={},
                 report_train_metrics=True,
                 lr=1e-4,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={},
                 **kwargs
                 ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.evaluation_quantile_bins = eval(evaluation_quantile_bins) if not isinstance(evaluation_quantile_bins, (list, ListConfig)) else evaluation_quantile_bins
        self.evaluation_time_points = eval(evaluation_time_points) if not isinstance(evaluation_time_points, (list, ListConfig)) else evaluation_time_points
        self.task_specific_exclusions = task_specific_exclusions
        self.task_weights = task_weights

        # set task names
        assert any([task_names, n_tasks]), 'Either provide number of tasks or task_names'
        if task_names and n_tasks:
            assert len(task_names) == n_tasks, 'Nr of task_names passed needs to match nr of tasks.'
            self.task_names = task_names
        elif n_tasks is not None and task_names is None:
            self.task_names = [f'event_{i}' for i in range(1, n_tasks + 1)]
        else:
            self.n_tasks = len(task_names)
            self.task_names = task_names

        # set events and event_names names:
        try:
            self.n_events = survival_task_kwargs['n_events'] #this needs to be set in the survival_taks_kwargs
        except KeyError:
            self.n_events = 1
            survival_task_kwargs['n_events'] = 1
        self.event_names = [f'event_{i}' for i in range(1, self.n_events+1)]

        # optimization
        self.lr = lr
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.schedule = schedule
        self.schedule_kwargs = schedule_kwargs

        # save the params.
        self.save_hyperparameters()

        # build heads
        self.netlist = [self.feature_extractor]
        self.heads = torch.nn.ModuleDict()
        self._build_heads()
        self.netlist.extend([m for m in self.heads.values()])

    def _build_heads(self):
        """
        Build Heads. Each Head is a independent survival taks with a shared basenet.
        :return:
        """
        if isinstance(self.hparams.survival_task, str):
            SurvivalTask = eval(self.hparams.survival_task)
        else:
            SurvivalTask = self.hparams.survival_task

        for task in self.task_names:
            self.heads[task] = SurvivalTask(network=deepcopy(self.hparams.latent_mlp),
                                            evaluation_quantile_bins=self.evaluation_quantile_bins,
                                            evaluation_time_points=self.evaluation_time_points,
                                            **self.hparams.survival_task_kwargs)
        # delete optimizer config (yes, I know I am being paranoid)
        for h in self.heads.values():
            h.configure_optimizers = None

    def configure_optimizers(self):
        if isinstance(self.optimizer, str):
            self.optimizer = eval(self.optimizer)
        if isinstance(self.schedule, str):
            self.schedule = eval(self.schedule)
        self.optimizer_kwargs["lr"] = self.lr

        params = []
        for net in self.netlist:
            for p in [net.parameters()]:
                params.extend(p)

        if self.schedule is not None:
            optimizer = self.optimizer(params, **self.optimizer_kwargs)
            if self.schedule == torch.optim.lr_scheduler.ReduceLROnPlateau:
                lr_scheduler = self.schedule(optimizer, **self.schedule_kwargs)
                scheduler = {'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}
                return [optimizer], [scheduler]
            else:
                return [optimizer], [self.schedule(optimizer,
                                                   **self.schedule_kwargs)]  # not working in 1.X ptl version with ReduceLROnPlateau
        else:
            return self.optimizer(params, **self.optimizer_kwargs)

    def unpack_batch(self, batch):
        if self.task_specific_exclusions:
            data, (durations, events) = batch
            *data, masks = data
            if len(data) == 1:
                data = data[0]
            return data, masks, durations, events
        else:
            data, (durations, events) = batch
            return data, durations, events

    def loss(self, args_dict):
        """
        Calculate Loss.

        :param args_dict: tuple, of len(self.heads), has the individual head's outputs of the shared_step method. The content of the tuple depends on the survival_task.
        :return:
        """
        loss = 0.
        loss_dict = dict()
        for task, args in args_dict.items():
            try:
                aux_dict = {}
                head_loss_dict = self.heads[task].loss(*args)
                loss += head_loss_dict['loss']
                for k in head_loss_dict.keys():
                    aux_dict[f"{task}_{k}"] = head_loss_dict[k]
                loss_dict.update(aux_dict)
            except:
                print(task, args)
        loss_dict['loss'] = loss
        return loss_dict

    @auto_move_data
    def forward(self, inputs, t=None):
        """
        :param inputs: tuple of tensors, should be (complex_data, covariates), the dim 0 needs to be shared between tensors.
        :param t: tensor, the timepoint at which to calculate the forward pass.
        :return:
        """
        # get fts
        features = self.feature_extractor(inputs)

        # run heads
        results = {}
        for task_name, task in self.heads.items():
            results[task_name] = task(features, t)

        return results

    def predict_risk(self, inputs, t=None):
        """

        :param inputs:
        :param t:
        :return:
        """
        preds = self.forward(inputs, t)
        for k in preds.keys():
            out = []
            for p_i in preds[k]:
                # print(p_i.shape, 'in')
                p = p_i.squeeze(-1)
                p = p.unsqueeze(0) if p.ndim < 2 else p
                # print(p.shape, 'out')
                out.append(p)
            preds[k] = out
            # preds[k] = tuple([p.squeeze(-1).unsqueeze(0) if p.ndim < 2 else p.squeeze(-1) for p in preds[k]])
        return preds

    def shared_step(self, data, duration, events):
        """
        Run a shared step through the network and all heads. Results are collected in a nested tuple of length self.heads.
        :param data: tuple, data for which to run shared_step. First element of tuple shall carry the complex data, second the covariates.
        :param duration: tensor, durations at which to calculate the loss, dim 0 is batchsize
        :param events: tensor, event indicator dim 0 is batchsize
        :return:
        """
        features = self.feature_extractor(data)

        # run heads
        results = {}
        for i, task in enumerate(self.heads.keys()):
            results[task] = self.heads[task].shared_step(features,
                                                         duration[:, i].unsqueeze(-1),
                                                         events[:, i].unsqueeze(-1)
                                                         )

        return results

    def apply_task_specific_exclusions(self, args_dict, masks):
        """
        Iterate over all elements of args_dict and apply exclusion mask
        :param data:
        :param masks:
        :return:
        """
        indicator = torch.Tensor([0])
        indicator = indicator.type_as(masks)

        for i, t in enumerate(args_dict.keys()):
            mask = torch.eq(masks[:, i], indicator)
            args = tuple([a[mask] for a in args_dict[t]])
            args_dict[t] = args

        return args_dict

    def training_step(self, batch, batch_idx, **kwargs):
        """
        Runs a training_step. See pl.LightningModule

        :param batch: tensor, data for 1 batch.
        :param batch_idx: int, index of batch.
        :return:
        """
        if self.task_specific_exclusions:
            data, masks, duration, events = self.unpack_batch(batch)
        else:
            data, duration, events = self.unpack_batch(batch)


        assert torch.all(data.isfinite())

        args_dict = self.shared_step(data, duration, events)

        if self.task_specific_exclusions:
            args_dict = self.apply_task_specific_exclusions(args_dict, masks)

        loss_dict = self.loss(args_dict)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx, **kwargs):
        """
        Runs a validation_step. See pl.LightningModule

        :param batch: tensor, data for 1 batch.
        :param batch_idx: int, index of batch.
        :return:
        """
        if self.task_specific_exclusions:
            data, masks, duration, events = self.unpack_batch(batch)
        else:
            data, duration, events = self.unpack_batch(batch)

        assert torch.all(data.isfinite())

        args_dict = self.shared_step(data, duration, events)

        durations = args_dict[self.task_names[0]][0]
        events = args_dict[self.task_names[0]][1]

        if self.hparams.num_testtime_views > 1:
            durations = durations[::self.hparams.num_testtime_views]
            events = events[::self.hparams.num_testtime_views]
            # do test time augmentation
            aggregated_args_dict = {}
            for head_name, args in args_dict.items():
                # first 2 are always duration and events, skip them.
                args_aggregated = []
                for a in args[2:]:
                    a = torch.cat([
                        torch.mean(j, dim=0).unsqueeze(0) for j in
                        torch.split(a, self.hparams.num_testtime_views, dim=0)])
                    args_aggregated.append(a)
                aggregated_args_dict[head_name] = ([durations, events] + args_aggregated)
            args_dict = aggregated_args_dict

        if self.task_specific_exclusions:
            args_dict = self.apply_task_specific_exclusions(args_dict, masks)

        loss_dict = self.loss(args_dict)
        loss_dict = dict([(f'val_{k}', v) for k, v in loss_dict.items()])
        for k, v in loss_dict.items():
            self.log(k, v, on_step=False, on_epoch=False, prog_bar=False, logger=False)
        return loss_dict

    def validation_epoch_end(self, outputs):
        metrics = {}
        # aggregate the per-batch-metrics:
        for metric_name in ["val_loss"]:
            metrics[metric_name] = torch.stack([output[metric_name] for output in outputs]).mean()

        # calculate the survival metrics
        valid_ds = self.val_dataloader().dataset if not \
            isinstance(self.val_dataloader().dataset, BatchedDS) \
            else self.val_dataloader().dataset.dataset
        train_ds = self.train_dataloader().dataset if not \
            isinstance(self.train_dataloader().dataset, BatchedDS) \
            else self.train_dataloader().dataset.dataset

        # valid metrics:
        print('Calculating VALID metrics')
        metrics_survival = self.calculate_survival_metrics(train_ds=train_ds, valid_ds=valid_ds,
                                                           time_points=self.evaluation_time_points,
                                                           quantile_bins=self.evaluation_quantile_bins)

        metrics.update(metrics_survival)

        # train metrics:
        if self.hparams.report_train_metrics:
            print('Calculating TRAIN metrics')
            train_metrics_survival = self.calculate_survival_metrics(train_ds=train_ds, valid_ds=train_ds,
                                                                     time_points=self.evaluation_time_points,
                                                                     quantile_bins=self.evaluation_quantile_bins)
            for key, value in train_metrics_survival.items():
                metrics[f'train_{key}'] = value

        for key, value in metrics.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def calculate_survival_metrics(self, train_ds, valid_ds, time_points=[5, 10], quantile_bins=None):
        """
        THIS IS THE COMPETING EVENTS VERSION!

        1. Calculate the Ctd on the quartiles of the valid set.
        2. Calculate the Brier scires for the same times.
        :return:
        """
        metrics = {}

        assert any([t in [time_points, quantile_bins] for t in ['None', 'none', None]]), 'EITHER pass quantiles OR pass timepoints'

        try:
            surv_train = np.stack([train_ds.events.values, train_ds.durations.values], axis=0)
            surv_valid = np.stack([valid_ds.events.values, valid_ds.durations.values], axis=0)
        except AttributeError:
            surv_train = np.stack([train_ds.events, train_ds.durations], axis=0)
            surv_valid = np.stack([valid_ds.events, valid_ds.durations], axis=0)

        if time_points is None:
            assert quantile_bins is not None, 'If timepoints is None, then pass quantile bins'
            taus = [np.quantile(surv_valid[1, surv_valid[0] > 0], q) for q in quantile_bins]
            annot = quantile_bins
        else:
            taus = time_points
            annot = time_points

        self.eval()
        loader = self.ext_dataloader(valid_ds, batch_size=5000, num_workers=4, shuffle=False, drop_last=False)
        with torch.no_grad():
            for i, tau in enumerate(taus):
                print(tau)
                risks_per_head = {tn: [] for tn in self.task_names}
                masks_per_head = {tn: [] for tn in self.task_names}
                tau_ = torch.Tensor([tau])
                for batch in loader:
                    if self.task_specific_exclusions:
                        data, m, d, e = self.unpack_batch(batch)
                    else:
                        data, d, e = self.unpack_batch(batch)
                        m = torch.zeros(size=(1, data.shape[0], len(self.task_names)))

                    # run subnet:
                    preds = self.predict_risk(data, t=tau_)
                    for j, (task, p) in enumerate(preds.items()):
                        risks_per_head[task].append(p[1].detach().cpu().numpy())  # 1 => Ft
                        masks_per_head[task].append(m[:, j].unsqueeze(0).cpu().numpy())  # 1 => Ft
                print('Done collecting.')

                for ti, task in enumerate(risks_per_head.keys()):
                    risks = np.concatenate(risks_per_head[task], axis=1)
                    masks = np.concatenate(masks_per_head[task], axis=1)
                    ctd_per_event = []
                    c_per_event = []
                    for e in range(1, self.hparams.n_events + 1):
                        try:
                            e_risks = risks[e - 1].ravel()
                            e_masks = masks[e-1].ravel()
                            e_risks = e_risks[e_masks < 1]
                            e_risks[pd.isna(e_risks)] = np.nanmax(e_risks)

                            # move to structured arrays:
                            struc_surv_train = np.array(
                                [(1 if e_ == e else 0, d) for e_, d in zip(surv_train[0][:, ti], surv_train[1][:, ti])],
                                dtype=[('event', 'bool'), ('duration', 'f8')])
                            struc_surv_valid = np.array(
                                [(1 if e_ == e else 0, d) for e_, d in zip(surv_valid[0][:, ti], surv_valid[1][:, ti][e_masks < 1])],
                                dtype=[('event', 'bool'), ('duration', 'f8')])

                            # np.any(struc_surv_valid["event"]):
                            Ctd = concordance_index_ipcw(struc_surv_train, struc_surv_valid,
                                                         e_risks,
                                                         tau=tau, tied_tol=1e-8)
                            C = concordance_index(event_times=surv_valid[1][:, ti][e_masks < 1],
                                                  predicted_scores=-e_risks,
                                                  event_observed=surv_valid[0][:, ti][e_masks < 1])
                        except: #ValueError: #else:
                            Ctd = [np.nan]
                            C = np.nan

                        ctd_per_event.append(Ctd[0])
                        c_per_event.append(C)
                        metrics[f'{task}__{self.event_names[e - 1]}_Ctd_{annot[i]}'] = Ctd[0]
                        metrics[f'{task}__{self.event_names[e - 1]}_C_{annot[i]}'] = C

                    metrics[f'{task}__Ctd_{annot[i]}'] = np.nanmean(np.asarray(ctd_per_event))
                    metrics[f'{task}__C_{annot[i]}'] = np.nanmean(np.asarray(c_per_event))

                for c in ['Ctd', 'C']:
                    aggr = []
                    for task in risks_per_head.keys():
                        aggr.append(metrics[f'{task}__{c}_{annot[i]}'])
                    metrics[f'Avg__{c}_{annot[i]}'] = np.nanmean(np.asarray(aggr))

        self.train()
        return metrics

    def ext_dataloader(self, ds, batch_size=None, shuffle=False, num_workers=None,
                       drop_last=False):  ### Already transformed datamodules? -> Or pass transformers?
        if batch_size is None:
            batch_size = self.batch_size
        if num_workers is None:
            num_workers = self.num_workers
        return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                          shuffle=shuffle, drop_last=drop_last)

    def predict_latents_dataset(self, ds, device='cuda:0'):
        # get a dataloader:
        loader = self.ext_dataloader(ds, batch_size=256, num_workers=4, shuffle=False, drop_last=False)

        latents = []

        for batch in loader:
            data, *_ = self.unpack_batch(batch)
            data = data.to(device)
            l = self.feature_extractor(data)
            latents.append(l.detach().cpu().numpy())

        latents = np.concatenate(latents, axis=0)
        latents_df = pd.DataFrame(latents, columns=[f'latent_{i}' for i in range(latents.shape[1])])
        del latents
        return latents_df

    def predict_dataset(self, dataset: object, times=[10]):
        """
        Predict dataset at times t
        :param ds:
        :param times:
        :return:
        """
        events, durations = [], []
        loader = self.ext_dataloader(dataset, batch_size=256, num_workers=4, shuffle=False, drop_last=False)

        self.eval()
        # predict each sample at each duration:
        dist_dict = {k: {tn: [] for tn in self.task_names} for k in ['ft', 'Ft', 'St']}

        with torch.no_grad():
            for batch in (loader):
                if self.task_specific_exclusions:
                    data, _, d, e = self.unpack_batch(batch)
                else:
                    data, d, e = self.unpack_batch(batch)
                durations.append(d)
                events.append(e)
                batch_preds_dict = {k: {task: [] for task in self.task_names} for k in ['ft', 'Ft', 'St']}
                for t in times:
                    preds = self.predict_risk(data,  t=torch.Tensor([t])) # preds is a dict w/ keys for each head
                    for task, v in preds.items():
                        batch_preds_dict['ft'][task].append(v[0].detach().cpu())
                        batch_preds_dict['Ft'][task].append(v[1].detach().cpu())
                        batch_preds_dict['St'][task].append(v[2].detach().cpu())
                del data

                for k in dist_dict.keys():
                    for task in dist_dict[k].keys():
                        dist_dict[k][task].append(torch.stack(batch_preds_dict[k][task], dim=1))

        for k in dist_dict.keys():
            for task in dist_dict[k].keys():
                dist_dict[k][task] = torch.cat(dist_dict[k][task], dim=-1).numpy()  # -> [e, t, n_samples]

        self.train()
        pred_df = []
        for task in self.heads.keys():
            head_pred_df = []
            for e in range(1, self.heads[task].n_events+1):
                t_df = []
                for t_i, t in enumerate(times):
                    df = pd.DataFrame.from_dict({
                        f"{e}_{t}_{k}__{task}": dist_dict[k][task][e-1, t_i, :].ravel() for k in dist_dict.keys()})
                    t_df.append(df)
                df = pd.concat(t_df, axis=1)
                head_pred_df.append(df)

            head_pred_df = pd.concat(head_pred_df, axis=1)
            pred_df.append(head_pred_df)
        pred_df = pd.concat(pred_df, axis=1)
        try:
            pred_df['durations'] = torch.cat(durations, dim=0).cpu().numpy().ravel()
            pred_df['events'] = torch.cat(events, dim=0).cpu().numpy().ravel()
        except ValueError: # happens when we have multiple events
            durations = torch.cat(durations, dim=0).cpu().numpy()
            events = torch.cat(events, dim=0).cpu().numpy()
            if durations.shape[-1] > 1: # more than 1 event => multitask => add duration cols individually
                for ti, tn in enumerate(self.task_names):
                    pred_df[f'durations_{tn}'] = durations[:, ti].ravel()
                    pred_df[f'events_{tn}'] = events[:, ti].ravel()
            else:
                raise ValueError()

        return pred_df

    def fit_isotonic_regressor(self, ds, times, n_samples):
        if len(ds) < n_samples:
            n_samples = len(ds)
        sample_idx = random.sample([i for i in range(len(ds))], n_samples)

        sample_ds = torch.utils.data.Subset(ds, sample_idx)
        pred_df = self.predict_dataset(sample_ds, np.array(times)).dropna()

        for h in self.heads.keys():

            if not hasattr(self, 'isoreg'):
                self.isoreg = {}

            for t in tqdm(times):
                if f"1_{t}_Ft__{h}" in self.isoreg:
                    continue
                else:
                    if pred_df[f"1_{t}_Ft__{h}"].nunique() > 1:
                        risk_obs = pd.Series(np.nan)
                        i = 1
                        while risk_obs.nunique() <= 1:
                            temp_df = pred_df.sample(frac=0.5, replace=True).dropna(subset=[f"1_{t}_Ft__{h}"])
                            try:
                                risk_obs, nan = get_observed_probability(temp_df[f"1_{t}_Ft__{h}"].values,
                                                                         temp_df[f"events_{h}"].values,
                                                                         temp_df[f"durations_{h}"].values,
                                                                         t)
                            except ConvergenceError:
                                if i == 20:
                                    break
                                i += 1
                        risk_pred = temp_df.drop(temp_df.index[nan])[f"1_{t}_Ft__{h}"].reset_index(drop=True)
                        self.isoreg[f"1_{t}_Ft__{h}"] = IsotonicRegression().fit(risk_pred.values, risk_obs.values)

    def predict_dataset_calibrated(self, ds, times):
        pred_df = self.predict_dataset(ds, np.array(times))
        for h in self.heads.keys():
            for t in times:
                pred_df[f"1_{t}_Ft__{h}__calibrated"] = self.isoreg[f"1_{t}_Ft__{h}"].predict(pred_df[f"1_{t}_Ft__{h}"])
        return pred_df


class DeepSurv(AbstractSurvivalTask):
    """
    Train a DeepSurv-Model
    """
    def __init__(self, network,
                 batch_size=128,
                 num_workers=8,
                 lr=1e-3,
                 evaluation_time_points=[10],
                 evaluation_quantile_bins=None,
                 report_train_metrics=True,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={},
                 **kwargs
                 ):
        """
        CoxPH Training in pl

        :param network:         `nn.Module` or `pl.LightningModule`, the network
        :param batch_size:      `int`, batchsize
        :param num_workers:     `int` nr of workers for the dataloader
        :param optimizer:       `torch.optim` class, the optimizer to apply
        :param optimizer_kwargs:    `dict` kwargs for optimizer
        :param schedule:        `LRschedule` class to use, optional
        :param schedule_kwargs: `dict` kwargs for scheduler
        """
        super().__init__(
            network=network,
            num_workers=num_workers,
            batch_size=batch_size,
            lr=lr,
            evaluation_time_points=evaluation_time_points,
            evaluation_quantile_bins=evaluation_quantile_bins,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            schedule=schedule,
            schedule_kwargs=schedule_kwargs
        )
        # save the params.
        self.save_hyperparameters()
        self.n_events = 1

    @property
    def __name__(self):
        return 'CoxPH'

    def shared_step(self, data, durations, events):
        logh = self.net(data)
        return durations, events, logh

    def loss(self, durations, events, logh):
        nll = cox_ph_loss(logh, durations, events)
        return {'loss': nll}

    # @auto_move_data
    def forward(self, X, t=None):
        """
        Predict a sample
        :return:
        """
        log_ph = self.net(X)
        f_t = log_ph
        F_t = log_ph
        S_t = 1 - log_ph
        return f_t, F_t, S_t

    @auto_move_data
    def predict_risk(self, X, t=None):
        """
        Predict RISK for a sample.
        :return:
    #   """
        log_ph = self.net(X)
        return log_ph

    def predict_dataset(self, dataset: object, times: object):
        """
        Predict the survival function for a sample at a given timepoint.
        :param dataset:
        :return:
        """
        log_hs = []
        durations = []
        events = []

        # get a loader for speed:
        loader = self.ext_dataloader(dataset, batch_size=256, num_workers=4, shuffle=False, drop_last=False)

        with torch.no_grad():
            for batch in loader:
                data, d, e = self.unpack_batch(batch)
                durations.append(d.cpu().detach())
                events.append(e.cpu().detach())
                log_hs.append(self.forward(data)[0].cpu().detach())
                del data
            del loader

        log_hs = torch.exp(torch.cat(log_hs), dim=0).numpy()

        pred_df = pd.DataFrame(log_hs.ravel(), columns=['loghs'])
        pred_df['durations'] = torch.cat(durations, dim=0).cpu().numpy().ravel()
        pred_df['events'] = torch.cat(events, dim=0).cpu().numpy().ravel()
        return pred_df

    def extract_latent(self, dataset: object):
        """
        Predict the survival function for a sample at a given timepoint.
        :param dataset:
        :return:
        """
        log_hs = []
        durations = []
        events = []

        # get a loader for speed:
        loader = self.ext_dataloader(dataset, batch_size=256, num_workers=4, shuffle=False, drop_last=False)

        with torch.no_grad():
            for batch in loader:
                data, d, e = self.unpack_batch(batch)
                durations.append(d.cpu().detach())
                events.append(e.cpu().detach())
                log_hs.append(self.forward(data)[0].cpu().detach())
                del data
            del loader

        log_hs = torch.exp(torch.cat(log_hs), dim=0).numpy()

        pred_df = pd.DataFrame(log_hs.ravel(), columns=['loghs'])
        # pred_df['cumbhs'] = cumbhs.ravel()
        pred_df['durations'] = torch.cat(durations, dim=0).cpu().numpy().ravel()
        pred_df['events'] = torch.cat(events, dim=0).cpu().numpy().ravel()
        return pred_df


class SoftSharingMultiTaskSurvivalTraining(MultiTaskSurvivalTraining):
    """
    MultiTaskSurvivalTraining with soft parameter sharing in the feature extractino nets.

    covariates, durations, events = sample  # here durations and events are multidimensional
    feature_extractor(covariates) -> latent

    preds = []
    for t in tasks:
        head_t(covariates) -> survival_pred_t
        preds.append(survival_pred_t)

    loss(survival_pred1,2,3 ; targets1,2,3)  # Mutlidimensional _NOT_ necessarily competing!
    """
    def __init__(self,
                 feature_extractor=None,
                 batch_size=128,
                 evaluation_time_points=[5, 10],
                 evaluation_quantile_bins=None,
                 task_specific_exclusions=None,
                 num_workers=8,
                 num_testtime_views=1,
                 latent_dim=100,
                 latent_mlp=None,
                 feature_dim=10,
                 task_names=None,
                 n_tasks=None,
                 survival_task=None,
                 survival_task_kwargs={},
                 report_train_metrics=True,
                 lr=1e-4,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={},
                 **kwargs
                 ):
        super().__init__(
            feature_extractor=feature_extractor,
            batch_size=batch_size,
            evaluation_time_points=evaluation_time_points,
            evaluation_quantile_bins=evaluation_quantile_bins,
            task_specific_exclusions=task_specific_exclusions,
            num_workers=num_workers,
            num_testtime_views=num_testtime_views,
            latent_dim=latent_dim,
            latent_mlp=latent_mlp,
            feature_dim=feature_dim,
            task_names=task_names,
            n_tasks=n_tasks,
            survival_task=survival_task,
            survival_task_kwargs=survival_task_kwargs,
            report_train_metrics=report_train_metrics,
            lr=lr,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            schedule=schedule,
            schedule_kwargs=schedule_kwargs,
            **kwargs

        )
        # build heads
        self.heads = torch.nn.ModuleDict()
        self._build_heads()
        self.feature_extractors = torch.nn.ModuleDict(
            [(h, deepcopy(self.feature_extractor)) for h in self.heads.keys()])
        self.netlist = [m for m in self.feature_extractors.values()]
        self.netlist.extend([m for m in self.heads.values()])

    def unpack_batch(self, batch):
        data, (durations, events) = batch
        return data, durations, events

    def soft_param_sharing_loss(self):
        """
        Reimplementation of the soft parameter sharing loss proposed here:
        https://gist.github.com/AvivNavon/fd9a98448bbf50352eaf8583fd36ec78

        :return:
        """
        param_groups = []
        for out in zip(*[net.named_parameters() for net in self.feature_extractors.values()]):
            if 'weight' in out[0][0]:
                param_groups.append([out[i][1] for i in range(len(out))])

        soft_sharing_loss = 0.
        for params in param_groups:
            params_soft_sharing_loss = 0.
            mean = torch.mean(torch.stack(params, axis=0), axis=0)
            for i in range(len(params)):
                params_soft_sharing_loss += torch.norm(mean - params[i], p='fro')
            soft_sharing_loss += params_soft_sharing_loss
        return soft_sharing_loss

    def loss(self, args_dict):
        """
        Calculate Loss.

        :param args_dict: tuple, of len(self.heads), has the individual head's outputs of the shared_step method. The content of the tuple depends on the survival_task.
        :return:
        """
        loss = 0.
        loss_dict = dict()
        for task, args in args_dict.items():
            aux_dict = {}
            head_loss_dict = self.heads[task].loss(*args)
            loss += head_loss_dict['loss']
            for k in head_loss_dict.keys():
                aux_dict[f"{task}_{k}"] = head_loss_dict[k]
            loss_dict.update(aux_dict)

        loss_dict['sharing_loss'] = self.soft_param_sharing_loss()
        loss_dict['loss'] = loss + loss_dict['sharing_loss']
        return loss_dict

    @auto_move_data
    def forward(self, inputs, t=None):
        """
        :param inputs: tuple of tensors, should be (complex_data, covariates), the dim 0 needs to be shared between tensors.
        :param t: tensor, the timepoint at which to calculate the forward pass.
        :return:
        """
        # run heads
        results = {}
        for task, net in self.heads.items():
            features = self.feature_extractors[task](inputs)
            results[task] = net(features, t)

        return results

    def shared_step(self, data, duration, events):
        """
        Run a shared step through the network and all heads. Results are collected in a nested tuple of length self.heads.
        :param data: tuple, data for which to run shared_step. First element of tuple shall carry the complex data, second the covariates.
        :param duration: tensor, durations at which to calculate the loss, dim 0 is batchsize
        :param events: tensor, event indicator dim 0 is batchsize
        :return:
        """
        # run heads
        results = {}
        for i, task in enumerate(self.heads.keys()):
            features = self.feature_extractors[task](data)
            results[task] = self.heads[task].shared_step(features,
                                                         duration[:, i].unsqueeze(-1),
                                                         events[:, i].unsqueeze(-1)
                                                         )

        return results


class ResidualMultiTaskSurvivalTraining(MultiTaskSurvivalTraining):
    """
    MultiTaskSurvivalTraining with Skip connection from covariates to head preds.

    covariates, durations, events = sample  # here durations and events are multidimensional
    feature_extractor(covariates) -> latent

    preds = []
    for t in tasks:
        head_t(covariates) -> survival_pred_t
        preds.append(survival_pred_t)

    loss(survival_pred1,2,3 ; targets1,2,3)  # Mutlidimensional _NOT_ necessarily competing!
    """
    def __init__(self,
                 feature_extractor=None,
                 batch_size=128,
                 evaluation_time_points=[5, 10],
                 evaluation_quantile_bins=None,
                 task_specific_exclusions=None,
                 num_workers=8,
                 num_testtime_views=1,
                 latent_dim=100,
                 latent_mlp=None,
                 feature_dim=10,
                 task_names=None,
                 n_tasks=None,
                 survival_task=None,
                 survival_task_kwargs={},
                 report_train_metrics=True,
                 lr=1e-4,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={},
                 **kwargs
                 ):
        super().__init__(
            feature_extractor=feature_extractor,
            batch_size=batch_size,
            evaluation_time_points=evaluation_time_points,
            evaluation_quantile_bins=evaluation_quantile_bins,
            task_specific_exclusions=task_specific_exclusions,
            num_workers=num_workers,
            num_testtime_views=num_testtime_views,
            latent_dim=latent_dim,
            latent_mlp=latent_mlp,
            feature_dim=feature_dim,
            task_names=task_names,
            n_tasks=n_tasks,
            survival_task=survival_task,
            survival_task_kwargs=survival_task_kwargs,
            report_train_metrics=report_train_metrics,
            lr=lr,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            schedule=schedule,
            schedule_kwargs=schedule_kwargs,
            **kwargs

        )

    @auto_move_data
    def forward(self, inputs, t=None):
        """
        :param inputs: tuple of tensors, should be (complex_data, covariates), the dim 0 needs to be shared between tensors.
        :param t: tensor, the timepoint at which to calculate the forward pass.
        :return:
        """
        # run shared net:
        features = self.feature_extractor(inputs)

        # run heads
        results = {}
        for task_name, net in self.heads.items():
            results[task_name] = net((features, inputs), t)

        return results

    def shared_step(self, data, duration, events):
        """
        Run a shared step through the network and all heads. Results are collected in a nested tuple of length self.heads.
        :param data: tuple, data for which to run shared_step. First element of tuple shall carry the complex data, second the covariates.
        :param duration: tensor, durations at which to calculate the loss, dim 0 is batchsize
        :param events: tensor, event indicator dim 0 is batchsize
        :return:
        """
        # run shared net:
        features = self.feature_extractor(data)

        # run heads
        results = {}
        for i, task in enumerate(self.heads.keys()):
            results[task] = self.heads[task].shared_step((features, data),
                                                         duration[:, i].unsqueeze(-1),
                                                         events[:, i].unsqueeze(-1)
                                                         )
        return results
