import blitz
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

from pycox.models.loss import nll_pmf_cr, rank_loss_deephit_cr
from pycox.models.utils import pad_col

from riskiano.source.losses.losses import *
from riskiano.source.evaluation.evaluation import get_observed_probability
from riskiano.source.datamodules.datasets import BatchedDS, DeepHitBatchedDS


class AbstractSurvivalTask(pl.LightningModule):
    """
    Defines a Task (in the sense of Lightning) to train a CoxPH-Model.
    """

    def __init__(self, network,
                 transforms=None,
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
        :param transforms:  `nn.Module or pl.LightningModule`, optional contains the Transforms applied to input.
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
        self.transforms = transforms

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
        if self.transforms is not None:
            data = self.transforms.apply_train_transform(data, events)
        args = self.shared_step(data, durations, events)
        loss_dict = self.loss(*args)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        data, durations, events = self.unpack_batch(batch)
        if self.transforms is not None:
            data = self.transforms.apply_valid_transform(data)
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
            isinstance(self.val_dataloader().dataset, (BatchedDS, DeepHitBatchedDS)) \
            else self.val_dataloader().dataset.dataset
        train_ds = self.train_dataloader().dataset if not \
            isinstance(self.train_dataloader().dataset, (BatchedDS, DeepHitBatchedDS)) \
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
                    if self.transforms is not None:
                        data = self.transforms.apply_valid_transform(data)
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


class AbstractCompetingRisksSurvivalTask(AbstractSurvivalTask):
    """
    ABC for competing risks tranings.
    """
    def __init__(self, network,
                 transforms=None,
                 n_events=1,
                 event_names=None,
                 batch_size=128,
                 num_workers=8,
                 evaluation_time_points=[5, 10],
                 evaluation_quantile_bins=None,
                 lr=1e-3,
                 report_train_metrics=True,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={},
                 **kwargs):
        """
        Abstract class for survival prediction with competing risks.

        :param network:
        :param transforms:
        :param n_events:    `int`, number of competing events. Minimum 1.
        :param event_names: `list`,  list of str, len() = n_events -> replaces names in logs and reported metricslist of str, same length as targets (=1) -> replaces names in logs and reported metrics.
        :param batch_size:
        :param num_workers:
        :param optimizer:
        :param optimizer_kwargs:
        :param schedule:    `Schedule class`,   will be instantiated by trainer.
        :param schedule_kwargs:  `dict`, schedule kwargs, like: {'patience':10, 'threshold':0.0001, 'min_lr':1e-6}
        """
        super().__init__(
            network=network,
            transforms=transforms,
            num_workers=num_workers,
            batch_size=batch_size,
            evaluation_time_points=evaluation_time_points,
            evaluation_quantile_bins=evaluation_quantile_bins,
            lr=lr,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            schedule=schedule,
            schedule_kwargs=schedule_kwargs)

        self.n_events = n_events
        if event_names is not None:
            assert len(event_names) == self.n_events, 'Nr of event_names passed needs to match nr of competing events.'
            self.event_names = event_names
        else:
            self.event_names = [f'event_{i}' for i in range(1, self.n_events+1)]

    @auto_move_data
    def predict_risk(self, X):
        """
        Predict __RISK__ for X.

        :param X:
        :return:
        """
        raise NotImplementedError()

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
            surv_train = np.stack([train_ds.events.values, train_ds.durations.values], axis=0).squeeze(axis=-1)
            surv_valid = np.stack([valid_ds.events.values, valid_ds.durations.values], axis=0).squeeze(axis=-1)
        except AttributeError:
            surv_train = np.stack([train_ds.events, train_ds.durations], axis=0).squeeze(axis=-1)
            surv_valid = np.stack([valid_ds.events, valid_ds.durations], axis=0).squeeze(axis=-1)

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
                    data, d, e = self.unpack_batch(batch)

                    if self.transforms is not None:
                        data = self.transforms.apply_valid_transform(data)
                    risk = self.predict_risk(data, t=tau_)
                    del data
                    risks.append(risk.detach().cpu().numpy())

            risks = np.concatenate(risks, axis=1)

            c_per_event = []
            ctd_per_event = []
            for e in range(1, self.n_events + 1):
                e_risks = risks[e-1].ravel()
                e_risks[pd.isna(e_risks)] = np.nanmax(e_risks)
                # move to structured arrays:
                struc_surv_train = np.array([(1 if e_ == e else 0, d) for e_, d in zip(surv_train[0], surv_train[1])],
                                            dtype=[('event', 'bool'), ('duration', 'f8')])
                struc_surv_valid = np.array([(1 if e_ == e else 0, d) for e_, d in zip(surv_valid[0], surv_valid[1])],
                                            dtype=[('event', 'bool'), ('duration', 'f8')])

                Ctd = concordance_index_ipcw(struc_surv_train, struc_surv_valid,
                                           e_risks,
                                           tau=tau, tied_tol=1e-8)
                C = concordance_index(event_times=surv_valid[1],
                                      predicted_scores=-e_risks,
                                      event_observed=surv_valid[0])
                ctd_per_event.append(Ctd[0])
                c_per_event.append(C)
                metrics[f'{self.event_names[e-1]}_Ctd_{annot[i]}'] = Ctd[0]
                metrics[f'{self.event_names[e-1]}_C_{annot[i]}'] = C

            metrics[f'Ctd_{annot[i]}'] = np.asarray(ctd_per_event).mean()
            metrics[f'C_{annot[i]}'] = np.asarray(c_per_event).mean()

        self.train()
        return metrics

    def fit_isotonic_regressor(self, ds, times, n_samples):
        if len(ds) < n_samples: n_samples = len(ds)
        sample_idx = random.sample([i for i in range(len(ds))], n_samples)
        sample_ds = torch.utils.data.Subset(ds, sample_idx)
        pred_df = self.predict_dataset(sample_ds, np.array(times)).dropna()
        for t in tqdm(times):
            if hasattr(self, 'isoreg'):
                if f"1_{t}_Ft" in self.isoreg:
                    pass
                else:
                    for i, array in enumerate([pred_df[f"1_{t}_Ft"].values, pred_df["events"].values, pred_df["durations"].values]):
                        if len(list(np.argwhere(np.isnan(array))))>0:
                            print(i)
                            print(np.argwhere(np.isnan(array)))
                    F_t_obs, nan = get_observed_probability(pred_df[f"1_{t}_Ft"].values, pred_df["events"].values, pred_df["durations"].values, t)
                    self.isoreg[f"1_{t}_Ft"] = IsotonicRegression().fit(pred_df.drop(
                        pred_df.index[nan])[f"1_{t}_Ft"].reset_index(drop=True).values, F_t_obs.values)
            else:
                F_t_obs, nan = get_observed_probability(pred_df[f"1_{t}_Ft"].values, pred_df["events"].values, pred_df["durations"].values, t)
                self.isoreg = {f"1_{t}_Ft": IsotonicRegression().fit(
                    pred_df.drop(pred_df.index[nan])[f"1_{t}_Ft"].reset_index(drop=True).values, F_t_obs.values)}

    def predict_dataset_calibrated(self, ds, times):
        pred_df = self.predict_dataset(ds, np.array(times))
        for t in times:
            pred_df[f"1_{t}_Ft__calibrated"] = self.isoreg[f"1_{t}_Ft"].predict(pred_df[f"1_{t}_Ft"])
        return pred_df

    def predict_dataset(self, ds, times):
        raise NotImplementedError('Abstract.')


class AbstractBayesianSurvivalTask(AbstractCompetingRisksSurvivalTask):
    """
    Abstract class providing the fns to do variational inference over the weighs.
    """
    def __init__(self,
                 network,
                 transforms=None,
                 batch_size=128,
                 evaluation_time_points=[5, 10],
                 evaluation_quantile_bins=None,
                 num_mc=10,
                 complexity_cost_weight=1e-6,
                 num_workers=8,
                 lr=1e-3,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={},
                 **kwargs
                 ):
        super().__init__(
            network=network, transforms=transforms, batch_size=batch_size,
            evaluation_time_points=evaluation_time_points, evaluation_quantile_bins=evaluation_quantile_bins,
            num_workers=num_workers, lr=lr, optimizer=optimizer, optimizer_kwargs=optimizer_kwargs, schedule=schedule,
            schedule_kwargs=schedule_kwargs)

        self.num_mc = num_mc
        self.complexity_cost_weight = complexity_cost_weight

    @property
    def __name__(self):
        return 'BayesAbstract'

    def sample_elbo(self, data, durations, events):
        """
        Here we need to sample the elbo over num_mc passes.
        :return:
        """
        loss = 0
        for _ in range(self.num_mc):
            args = self.shared_step(data, durations, events)
            loss += self.loss(*args)['loss']
            for net in self.networks:
                try:
                    loss += net.nn_kl_divergence() * self.complexity_cost_weight
                except AttributeError:
                    continue

        return loss / self.num_mc

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

    def shared_step(self, data, durations, events):
        """
        shared step for the SQR model.

        :param data:
        :param duration:
        :param events:
        :return:
        """
        predictions = self.net(data)
        return predictions, durations, events

    def training_step(self, batch, batch_idx):
        data, durations, events = self.unpack_batch(batch)
        if self.transforms is not None:
            data = self.transforms.apply_train_transform(data, events)
        loss = self.sample_elbo(data, durations, events)
        # add the other losses here...
        #
        #
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, durations, events = self.unpack_batch(batch)
        if self.transforms is not None:
            data = self.transforms.apply_valid_transform(data)
        args = self.shared_step(data, durations, events)
        loss_dict = self.loss(*args)
        return dict([(f'val_{k}', v) for k, v in loss_dict.items()])

    def predict_risk(self, X):
        """
        Helper function to calculate survival metrics.
        This function should ONLY return the risk estimate to calculate C-Idx.
        :return:
        """
        preds, _ = self.sample_forward(X)
        return preds

    def sample_forward(self, X, n_samples=100, forward_kwargs={}):
        """
        Sample Preds.
        :param n_samples: `int`, n_samples
        :return:
        """
        preds = [self.forward(X, **forward_kwargs) for _ in range(n_samples)]
        preds = torch.stack([p[0] for p in preds], dim=0)
        std = preds.std(dim=0)
        preds = preds.mean(dim=0)
        return preds, std

    def predict_dataset(self, dataset: object, times=[10]):
        """
        Predict dataset with Uncertainty!

        :param dataset:
        :param times:
        :return:
        """
        pred_means, pred_stds = [], []
        durations, events = [], []

        # get a loader for speed:
        loader = self.ext_dataloader(dataset, batch_size=256, num_workers=4, shuffle=False, drop_last=False)

        with torch.no_grad():
            for batch in loader:
                data, d, e = self.unpack_batch(batch)
                if self.transforms is not None:
                    data = self.transforms.apply_valid_transform(data)
                durations.append(d.cpu().detach())
                events.append(e.cpu().detach())
                pred, pred_std = self.sample_forward(data)
                pred_means.append(pred.cpu().detach())
                pred_stds.append(pred_std.cpu().detach())
                del data
            del loader
        pred_means = torch.cat(pred_means, dim=0).numpy().ravel()
        pred_stds = torch.cat(pred_stds, dim=0).numpy().ravel()
        durations = torch.cat(durations, dim=0).squeeze().numpy().ravel()
        events = torch.cat(events, dim=0).squeeze().numpy().ravel()

        pred_df = pd.DataFrame.from_dict({"prediction_mean": pred_means,
                                          "prediction_stds": pred_stds,
                                          "durations": durations,
                                          "events": events})
        return pred_df

    def predict_dataset_calibrated(self, ds, times):
        pred_df = self.predict_dataset(ds, np.array(times))
        for t in times:
            pred_df[f"1_{t}_Ft__calibrated"] = self.isoreg[f"1_{t}_Ft"].predict(pred_df[f"1_{t}_Ft"])
            pred_df[f"1_{t}_stds__calibrated"] = pred_df[f"1_{t}_Ft__calibrated"]/pred_df[f"1_{t}_Ft"] * pred_df[f"1_{t}_Ft_stds"]
        return pred_df


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
                 transforms=None,
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
        self.transforms = transforms
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

        if self.transforms is not None:
            data = self.transforms.apply_train_transform(data)
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

        if self.transforms is not None:
            data = self.transforms.apply_valid_transform(data)
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
            isinstance(self.val_dataloader().dataset, (BatchedDS, DeepHitBatchedDS)) \
            else self.val_dataloader().dataset.dataset
        train_ds = self.train_dataloader().dataset if not \
            isinstance(self.train_dataloader().dataset, (BatchedDS, DeepHitBatchedDS)) \
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

                    if self.transforms is not None:
                        data = self.transforms.apply_valid_transform(data)

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
            if self.transforms is not None:
                data = self.transforms.apply_valid_transform(data)

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
                if self.transforms is not None:
                    data = self.transforms.apply_valid_transform(data)
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


class MultiHeadSurvivalTraining(MultiTaskSurvivalTraining):
    def __init__(self,
                 complex_feature_extractor=None,
                 feature_extractor=None,
                 latent_mlp_dict={'covariates': nn.Identity(),
                                  'latent': nn.Identity(),
                                  'combined': nn.Identity(),
                                  },
                 transforms=None,
                 batch_size=128,
                 evaluation_time_points=[10],
                 evaluation_quantile_bins=None,
                 num_workers=8,
                 num_testtime_views=1,
                 latent_dim=100,
                 feature_dim=10,
                 survival_task=None,
                 survival_task_kwargs={},
                 lr=1e-4,
                 report_train_metrics=True,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={},
                 **kwargs
                 ):
        super(MultiTaskSurvivalTraining, self).__init__(
            evaluation_time_points=evaluation_time_points,
            evaluation_quantile_bin=evaluation_quantile_bins
        )
        self.complex_feature_extractor = complex_feature_extractor
        self.feature_extractor = feature_extractor
        self.transforms = transforms

        self.task_names = ['covariates', 'combined', 'latent']

        # optimization
        self.lr = lr
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.schedule = schedule
        self.schedule_kwargs = schedule_kwargs

        # save the params.
        self.save_hyperparameters()

        try:
            self.n_events = survival_task_kwargs['n_events'] #this needs to be set in the survival_taks_kwargs
        except KeyError:
            self.n_events = 1
            survival_task_kwargs['n_events'] = 1
        self.event_names = [f'event_{i}' for i in range(1, self.n_events+1)]

        # build heads
        self.netlist = [self.feature_extractor, self.complex_feature_extractor]
        self.heads = torch.nn.ModuleDict()
        self._build_heads(latent_mlp_dict)
        self.netlist.extend([m for m in self.heads.values()])

    def _build_heads(self, latent_mlp_dict):
        """
        Build Heads. Each Head is a independent survival taks with a shared basenet.
        :return:
        """

        self.heads['covariates'] = self.hparams.survival_task(network=latent_mlp_dict.pop('covariates', nn.Identity()),
                                                              output_dim=self.hparams.latent_dim,
                                                              **self.hparams.survival_task_kwargs)
        self.heads['combined'] = self.hparams.survival_task(network=latent_mlp_dict.pop('combined', nn.Identity()),
                                                            # output_dim=100,
                                                            output_dim=2*self.hparams.latent_dim,
                                                            **self.hparams.survival_task_kwargs)
        self.heads['latent'] = self.hparams.survival_task(network=latent_mlp_dict.pop('latent', nn.Identity()),
                                                          output_dim=self.hparams.latent_dim,
                                                          **self.hparams.survival_task_kwargs)
        # delete optimizer config (yes, I know I am being paranoid)
        for h in self.heads.values():
            h.configure_optimizers = None

    @auto_move_data
    def forward(self, inputs, t=None):
        """
        :param inputs: tuple of tensors, should be (complex_data, covariates), the dim 0 needs to be shared between tensors.
        :param t: tensor, the timepoint at which to calculate the forward pass.
        :return:
        """
        # unpack
        complex_data, covariates = inputs

        # get fts
        complex_latent = self.complex_feature_extractor(complex_data)

        covariates_latent = self.feature_extractor(covariates)

        # run heads
        results = {}
        results['covariates'] = self.heads['covariates'](covariates_latent, t)
        results['combined'] = self.heads['combined'](torch.cat([covariates_latent.detach(), complex_latent], dim=-1), t)
        results['latent'] = self.heads['latent'](complex_latent.detach(), t)

        return results

    @auto_move_data
    def shared_step(self, data, duration, events):
        """
        Run a shared step through the network and all heads. Results are collected in a nested tuple of length self.heads.
        :param data: tuple, data for which to run shared_step. First element of tuple shall carry the complex data, second the covariates.
        :param duration: tensor, durations at which to calculate the loss, dim 0 is batchsize
        :param events: tensor, event indicator dim 0 is batchsize
        :return:
        """
        # unpack
        complex_data, covariates = data

        complex_latent = self.complex_feature_extractor(complex_data)
        covariates_latent = self.feature_extractor(covariates)

        # run heads
        results = {}
        results['covariates'] = self.heads['covariates'].shared_step(covariates_latent, duration, events)
        results['combined'] = self.heads['combined'].shared_step(torch.cat([covariates_latent.detach(), complex_latent],
                                                                           dim=-1),
                                                                 duration, events)
        results['latent'] = self.heads['latent'].shared_step(complex_latent.detach(), duration, events)

        return results

    def calculate_survival_metrics(self, train_ds, valid_ds, time_points=[10], quantile_bins=None):
        """
        THIS IS THE COMPETING EVENTS VERSION!

        1. Calculate the Ctd on the quartiles of the valid set.
        2. Calculate the Brier scires for the same times.
        :return:
        """
        metrics = {}

        assert None in [time_points, quantile_bins], 'EITHER pass quantiles OR pass timepoints'

        try:
            surv_train = np.stack([train_ds.events.values, train_ds.durations.values], axis=0).squeeze(axis=-1)
            surv_valid = np.stack([valid_ds.events.values, valid_ds.durations.values], axis=0).squeeze(axis=-1)
        except AttributeError:
            surv_train = np.stack([train_ds.events, train_ds.durations], axis=0).squeeze(axis=-1)
            surv_valid = np.stack([valid_ds.events, valid_ds.durations], axis=0).squeeze(axis=-1)


        if time_points is None:
            assert quantile_bins is not None, 'If timepoints is None, then pass quantile bins'
            taus = [np.quantile(surv_valid[1, surv_valid[0] > 0], q) for q in quantile_bins]
            annot = quantile_bins
        else:
            taus = time_points
            annot = time_points

        self.eval()
        loader = self.ext_dataloader(valid_ds, batch_size=256, num_workers=4, shuffle=False, drop_last=False)
        with torch.no_grad():
            for i, tau in enumerate(taus):
                risks_per_head = {tn: [] for tn in self.task_names}
                tau_ = torch.Tensor([tau])
                for batch in loader:
                    data, d, e = self.unpack_batch(batch)

                    if self.transforms is not None:
                        data = self.transforms.apply_valid_transform(data)

                    # run subnet:
                    preds = self.predict_risk(data, t=tau_)
                    for task, p in preds.items():
                        risks_per_head[task].append(p.detach().cpu().numpy())

                for task in risks_per_head.keys():
                    risks = np.concatenate(risks_per_head[task], axis=1)
                    ctd_per_event = []
                    c_per_event = []
                    for e in range(1, self.hparams.n_events + 1):
                        e_risks = risks[e - 1].ravel()
                        e_risks[pd.isna(e_risks)] = np.nanmax(e_risks)
                        # move to structured arrays:
                        struc_surv_train = np.array(
                            [(1 if e_ == e else 0, d) for e_, d in zip(surv_train[0], surv_train[1])],
                            dtype=[('event', 'bool'), ('duration', 'f8')])
                        struc_surv_valid = np.array(
                            [(1 if e_ == e else 0, d) for e_, d in zip(surv_valid[0], surv_valid[1])],
                            dtype=[('event', 'bool'), ('duration', 'f8')])

                        Ctd = concordance_index_ipcw(struc_surv_train, struc_surv_valid,
                                                     e_risks,
                                                     tau=tau, tied_tol=1e-8)
                        C = concordance_index(event_times=surv_valid[1],
                                              predicted_scores=-risks,
                                              event_observed=surv_valid[0])

                        ctd_per_event.append(Ctd[0])
                        c_per_event.append(C)
                        metrics[f'{task}__{self.event_names[e - 1]}_Ctd_{annot[i]}'] = Ctd[0]
                        metrics[f'{task}__{self.event_names[e - 1]}_C_{annot[i]}'] = C

                    metrics[f'{task}__Ctd_{annot[i]}'] = np.asarray(ctd_per_event).mean()
                    metrics[f'{task}__C_{annot[i]}'] = np.asarray(c_per_event).mean()

        self.train()
        return metrics



class DeepSurv(AbstractSurvivalTask):
    """
    Train a DeepSurv-Model
    """
    def __init__(self, network,
                 transforms=None,
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
        :param transforms:      `nn.Module` holding the transformations to apply to datamodules if necessary
        :param batch_size:      `int`, batchsize
        :param num_workers:     `int` nr of workers for the dataloader
        :param optimizer:       `torch.optim` class, the optimizer to apply
        :param optimizer_kwargs:    `dict` kwargs for optimizer
        :param schedule:        `LRschedule` class to use, optional
        :param schedule_kwargs: `dict` kwargs for scheduler
        """
        super().__init__(
            network=network,
            transforms=transforms,
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

    # def on_save_checkpoint(self, checkpoint):
    #     self.bhs, _ = self.compute_bhs(dataloader=self.train_dataloader())
    #     checkpoint["bhs"] = self.bhs
    #
    # def on_load_checkpoint(self, checkpoint):
    #     self.bhs = checkpoint["bhs"]
    #
    # def compute_bhs(self, dataloader):
    #     events = []
    #     durations = []
    #     logh = []
    #
    #     max_duration = dataloader.dataset.durations.flatten().max()  # currently UKBB
    #     with torch.no_grad():
    #         for batch in dataloader:
    #             data, (d, e) = batch
    #             if self.transforms is not None:
    #                 data = self.transforms.apply_valid_transform(data)
    #             durations.append(d.cpu().detach())
    #             events.append(e.cpu().detach())
    #             logh.append(self.forward(data)[0].cpu().detach())
    #
    #     logh = torch.cat(logh, dim=0).numpy().ravel()
    #     durations = torch.cat(durations, dim=0).numpy().ravel()
    #     events = torch.cat(events, dim=0).numpy().ravel()
    #
    #     x = np.array([np.exp(logh), durations, events])
    #     df = pd.DataFrame({'expg': x[0], 'durations': x[1], 'events': x[2]})
    #     df.sort_values(by='durations', ascending=False)
    #
    #     bhs = (df
    #                .groupby('durations')
    #                .agg({'expg': 'sum', 'events': 'sum'})
    #                .sort_index(ascending=False)
    #                .assign(expg=lambda x: x['expg'].cumsum())
    #                .pipe(lambda x: x['events'] / x['expg'])
    #                .fillna(0.)
    #                .iloc[::-1]
    #                .loc[lambda x: x.index <= max_duration]
    #                )
    #
    #     bhs = bhs.to_frame()
    #     bhs.columns = ['baseline_hazards']
    #     return bhs, durations

    def predict_dataset(self, dataset: object, times: object):
        """
        Predict the survival function for a sample at a given timepoint.
        :param dataset:
        :return:
        """
        # bhs = self.bhs

        log_hs = []
        durations = []
        events = []

        # get a loader for speed:
        loader = self.ext_dataloader(dataset, batch_size=256, num_workers=4, shuffle=False, drop_last=False)

        with torch.no_grad():
            for batch in loader:
                data, d, e = self.unpack_batch(batch)
                if self.transforms is not None:
                    data = self.transforms.apply_valid_transform(data)
                durations.append(d.cpu().detach())
                events.append(e.cpu().detach())
                log_hs.append(self.forward(data)[0].cpu().detach())
                del data
            del loader

        log_hs = torch.exp(torch.cat(log_hs), dim=0).numpy()

        # bhs['cumbhs'] = np.cumsum(bhs['baseline_hazards'].values)
        # cumbhs = np.expand_dims(bhs['cumbhs'].values, axis=-1)
        #
        # F_t_all = pd.DataFrame(np.matmul(cumbhs, log_hs.T),
        #                        index=[i for i in bhs.index.values])  # [n_durations x n_samples]
        #
        # F_tX = []
        # for t in times.ravel():
        #     F_tX.append(F_t_all.iloc[[F_t_all.index.get_loc(t, method="nearest")]].values)
        # F_tX = np.concatenate(F_tX, axis=0)  # [times, n_samples]
        # S_tX = 1 - F_tX
        #
        # pred_df = []
        # for t_i, t in enumerate(times):
        #     df = pd.DataFrame.from_dict({
        #         f"1_{t}_Ft": F_tX[t_i, :].ravel(),
        #         f"1_{t}_St": S_tX[t_i, :].ravel()})
        # pred_df.append(df)

        # pred_df = pd.concat(pred_df, axis=1)
        # pred_df['loghs'] = log_hs.ravel()

        pred_df = pd.DataFrame(log_hs.ravel(), columns=['loghs'])
        # pred_df['cumbhs'] = cumbhs.ravel()
        pred_df['durations'] = torch.cat(durations, dim=0).cpu().numpy().ravel()
        pred_df['events'] = torch.cat(events, dim=0).cpu().numpy().ravel()
        return pred_df

    def extract_latent(self, dataset: object):
        """
        Predict the survival function for a sample at a given timepoint.
        :param dataset:
        :return:
        """
        # bhs = self.bhs

        log_hs = []
        durations = []
        events = []

        # get a loader for speed:
        loader = self.ext_dataloader(dataset, batch_size=256, num_workers=4, shuffle=False, drop_last=False)

        with torch.no_grad():
            for batch in loader:
                data, d, e = self.unpack_batch(batch)
                if self.transforms is not None:
                    data = self.transforms.apply_valid_transform(data)
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


class DeepSurvivalMachine(AbstractCompetingRisksSurvivalTask):
    """
    pl.Lightning Module for DeepSurvivalMachines.
    """
    def __init__(self,
                 network=None,
                 transforms=None,
                 n_events=1,
                 event_names=None,
                 batch_size=1024,
                 alpha=1,
                 gamma=1e-8,
                 k_dim=8,
                 output_dim=100,
                 temperature=1000,
                 network_shape=None,
                 network_scale=None,
                 network_ks=None,
                 distribution='weibull',
                 num_workers=8,
                 lr=1e-2,
                 report_train_metrics=True,
                 evaluation_time_points=[10],
                 evaluation_quantile_bins=None,
                 optimizer=torch.optim.SGD,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={},
                 lambda_sparse=None,
                 **kwargs
                 ):

        """
        pl.Lightning Module for DeepSurvivalMachines.
        :param network:          `nn.Module` or `pl.LightningModule`, the network
        :param transforms:       `nn.Module` holding the transformations to apply to data if necessary
        :param n_events:         `int`, number of events to consider. Minimum 1.
        :param batch_size:       `int`, batchsize
        :param alpha:            `float`, ]0, 1] weight for the loss function (1 = equal ratio, <1, upweighting of f_t(X))
        :param gamma:            `float`, factor to add to the shape and scale params to avoid edge conditions.
        :param k_dim:            `int`, number of distributions in the mixture
        :param output_dim:       `int`, outdim of `network` and in_dim to the layers generating k and `
        :param temperature:      `int`, temperature of the softmax parameter
        :param network_scale:    `nn.Module` or `pl.LightningModule`, the network to be used to compute the scale param,
                                    optional, if None, will put in linear layer.
        :param network_shape:    `nn.Module` or `pl.LightningModule`, the network to be used to compute the shape param,
                                    optional, if None, will put in linear layer.
        :param network_ks:       `nn.Module` or `pl.LightningModule`, the network to be used to compute the ks param,
                                    optional, if None, will put in linear layer.
        :param distribution:     `str` [`weibull`, `lognormal`] the base distribution
        :param num_workers:      `int` nr of workers for the dataloader
        :param optimizer:        `torch.optim` class, the optimizer to apply
        :param optimizer_kwargs: `dict` kwargs for optimizer
        :param schedule:         `LRschedule` class to use, optional
        :param schedule_kwargs:  `dict` kwargs for scheduler
        :param lambda_sparse:    `float` 1e-3 or lower; multiplier for the sparsity loss when using tabnet
        """
        if network is None:
            raise ValueError('You need to pass a network.')
        super().__init__(
            network=network,
            transforms=transforms,
            n_events=n_events,
            num_workers=num_workers,
            batch_size=batch_size,
            lr=lr,
            evaluation_time_points=evaluation_time_points,
            evaluation_quantile_bins=evaluation_quantile_bins,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            schedule=schedule,
            schedule_kwargs=schedule_kwargs)

        self.lambda_sparse = lambda_sparse
        self.temperature = temperature
        self.output_dim = output_dim
        self.k_dim = k_dim
        self.alpha = alpha
        self.gamma = gamma
        assert distribution in ['weibull', 'lognormal'], 'Currently only `lognormal` & `weibull` available.'
        self.distribution = distribution

        # build the nets:
        self.scale = network_scale.to(self.device) if network_scale is not None else \
            nn.Sequential(nn.Linear(self.output_dim, self.k_dim * self.n_events, bias=True),
                          nn.Softplus()
                          ).to(self.device)
        self.shape = network_shape.to(self.device) if network_shape is not None else \
            nn.Sequential(nn.Linear(self.output_dim, self.k_dim * self.n_events, bias=True),
                               nn.Softplus()
                               ).to(self.device)
        self.ks = network_ks.to(self.device) if network_ks is not None else\
            nn.Linear(self.output_dim, self.k_dim * self.n_events, bias=True).to(self.device)

        # set params to be optimized:
        self.params = []
        self.networks = [self.net, self.scale, self.shape, self.ks]
        for n in self.networks:
            if n is not None:
                self.params.extend(list(n.parameters()))

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        self.save_hyperparameters()

    @property
    def __name__(self):
        return 'DeepSurvivalMachine'

    @auto_move_data
    def sample_eventtime_from_mixture(self, data, nsamples=100):
        """
        Sample event time from mixture.
        :param scale:
        :param shape:
        :return:
        """

        if self.net.__class__.__name__ == 'TabNet':
            features, M_loss = self.net(data)
        else:
            features = self.net(data)

        scale = self.gamma + self.scale(features).view(features.size(0), self.k_dim, -1)
        shape = self.gamma + self.shape(features).view(features.size(0), self.k_dim, -1)
        ks = self.ks(features).view(features.size(0), self.k_dim, -1) / self.temperature
        # get dists
        if self.distribution == 'lognormal':
            distribution = LogNormal(
                loc=shape, # loc (float or Tensor) – mean of log of distribution
                scale=scale, # scale (float or Tensor) – standard deviation of log of the distribution
                validate_args=True)
        elif self.distribution == 'weibull':
            distribution = Weibull(
                scale,
                shape,
                validate_args=True
            )
        else:
            raise NotImplementedError('Currently only `lognormal` & `weibull` available.')

        samples = [distribution.sample() for _ in range(nsamples)]

        samples = torch.stack(samples, axis=0) # [samples, B, k, e]
        # weighted mean
        samples = (samples * F.softmax(ks, dim=1).repeat(nsamples, 1,1,1)).sum(axis=2) / self.k_dim
        sample_stds = samples.std(axis=0) # 1, B, k
        sample_means = samples.mean(axis=0)

        return sample_means, sample_stds

    def calculate_loglikelihood_under_mixture(self, scale, shape, durations):
        """Sample from the distribution"""
        if durations.dim() < 3:
            durations = durations.unsqueeze(-1)

        try:
            if self.distribution == 'lognormal':
                distribution = LogNormal(
                    loc=shape, # loc (float or Tensor) – mean of log of distribution
                    scale=scale, # scale (float or Tensor) – standard deviation of log of the distribution
                    validate_args=True)
                logf_t = distribution.log_prob(durations)
                logF_t = torch.log(0.5 + 0.5 * torch.erf(torch.div(torch.log(durations) - shape,
                                                                   np.sqrt(2)*scale)))
                logS_t = torch.log(0.5 - 0.5 * torch.erf(torch.div(torch.log(durations) - shape,
                                                                   np.sqrt(2)*scale)))
            elif self.distribution == 'weibull':
                distribution = Weibull(
                    scale,
                    shape,
                    validate_args=True
                )
                logf_t = distribution.log_prob(durations)
                logF_t = torch.log(1-torch.exp(-torch.pow(durations.div(scale), shape)))
                logS_t = -torch.pow(durations.div(scale), shape)
            else:
                raise NotImplementedError('Currently only `lognormal` & `weibull` available.')
        except ValueError:
            raise KeyboardInterrupt('NaNs in params, aborting training.')

        return logf_t, logF_t, logS_t

    def shared_step(self, data, durations, events):
        # TabNet returns a tuple
        if self.net.__class__.__name__ == 'TabNet':
            features, M_loss = self.net(data)
        else:
            features = self.net(data)

        scale = self.gamma + self.scale(features).view(features.size(0), self.k_dim, -1)
        shape = self.gamma + self.shape(features).view(features.size(0), self.k_dim, -1)

        ks = self.ks(features).view(features.size(0), self.k_dim, -1) / self.temperature

        logf_t, logF_t, logS_t = self.calculate_loglikelihood_under_mixture(scale, shape, durations)

        if self.net.__class__.__name__ is 'TabNet':
            return durations, events, logf_t, logF_t, logS_t, scale, shape, ks, M_loss
        else:
            return durations, events, logf_t, logF_t, logS_t, scale, shape, ks

    def loss(self, durations, events, logf_t, logF_t, logS_t, scale, shape, ks, M_loss=None):
        """ Calculate total DSM loss."""
        elbo_u = 0.
        elbo_c = 0.

        for e in range(1, self.n_events+1):
            elbo_u += DSM_uncensored_loss(logf_t[:, :, e-1], ks[:, :, e-1], events, e=e)
            elbo_c += DSM_censored_loss(logS_t[:, :, e-1], ks[:, :, e-1], events, e=e)

        loss_val = elbo_u + self.alpha * elbo_c

        # Sparsity loss multiplier for TabNet
        if self.net.__class__.__name__ is 'TabNet' and self.lambda_sparse is not None:
            loss_val = loss_val - (self.lambda_sparse * M_loss)

        return {'loss': loss_val,
                'uc_loss': elbo_u,
                'c_loss': elbo_c,
                }

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
        f_tX, F_tX, S_tX, t_X, t_X_std = [], [], [], [], []

        with torch.no_grad():
            for batch in (loader):
                data, d, e = self.unpack_batch(batch)
                if self.transforms is not None:
                    data = self.transforms.apply_valid_transform(data)
                durations.append(d)
                events.append(e)
                f_sample, F_sample, S_sample = [], [], []
                for t in times:
                    f_preds, F_preds, S_preds = self.forward(data,  t=torch.Tensor([t]))
                    f_sample.append(f_preds.detach().cpu())
                    F_sample.append(F_preds.detach().cpu())
                    S_sample.append(S_preds.detach().cpu())

                # sample the event time (argmax f(t))
                t_sample, t_sample_std = self.sample_eventtime_from_mixture(data) # [B, e]
                del data
                t_X.append(t_sample.permute(1, 0).detach().cpu())
                t_X_std.append(t_sample_std.permute(1, 0).detach().cpu())
                S_tX.append(torch.stack(S_sample, dim=1))
                F_tX.append(torch.stack(F_sample, dim=1))
                f_tX.append(torch.stack(f_sample, dim=1))
        t_X = torch.cat(t_X, dim=-1).numpy()  # -> [e, t, n_samples]
        t_X_std = torch.cat(t_X_std, dim=-1).numpy()  # -> [e, t, n_samples]
        S_tX = torch.cat(S_tX, dim=-1).numpy()  # -> [e, t, n_samples]
        F_tX = torch.cat(F_tX, dim=-1).numpy()
        f_tX = torch.cat(f_tX, dim=-1).numpy()

        self.train()
        pred_df = []
        for e in range(1, self.n_events+1):
            t_df = []
            for t_i, t in enumerate(times):
                df = pd.DataFrame.from_dict({
                    f"{e}_{t}_ft": f_tX[e-1, t_i, :].ravel(),
                    f"{e}_{t}_Ft": F_tX[e-1, t_i, :].ravel(),
                    f"{e}_{t}_St": S_tX[e-1, t_i, :].ravel()})
                t_df.append(df)
            df = pd.concat(t_df, axis=1)
            df[f'{e}_time'] = t_X[e-1, :].ravel()
            df[f'{e}_time_std'] = t_X_std[e-1, :].ravel()
            pred_df.append(df)

        pred_df = pd.concat(pred_df, axis=1)
        pred_df['durations'] = torch.cat(durations, dim=0).cpu().numpy().ravel()
        pred_df['events'] = torch.cat(events, dim=0).cpu().numpy().ravel()
        return pred_df

    @auto_move_data
    def forward(self, data, t=None):
        # TabNet returns the output and
        if self.net.__class__.__name__ is 'TabNet':
            features, M_loss = self.net(data)
        else:
            features = self.net(data)

        batchsize = features.size(0)
        if batchsize != t.size(0):
            t = t.repeat(batchsize, 1)

        scale = self.gamma + self.scale(features).view(batchsize, self.k_dim, -1)
        shape = self.gamma + self.shape(features).view(batchsize, self.k_dim, -1)

        ks = self.ks(features).view(batchsize, self.k_dim, -1) / self.temperature

        logf_t, logF_t, logS_t = self.calculate_loglikelihood_under_mixture(scale, shape, t)

        S_t = torch.exp(torch.logsumexp(logS_t.add(F.log_softmax(ks, dim=1)), 1, keepdim=True)).permute(2, 1, 0).squeeze(1)
        F_t = torch.exp(torch.logsumexp(logF_t.add(F.log_softmax(ks, dim=1)), 1, keepdim=True)).permute(2, 1, 0).squeeze(1)
        f_t = torch.exp(torch.logsumexp(logf_t.add(F.log_softmax(ks, dim=1)), 1, keepdim=True)).permute(2, 1, 0).squeeze(1)

        return f_t, F_t, S_t

    @auto_move_data
    def predict_risk(self, X, t=None):
        """
        Predict Risk (= F(t)) nothing else.
        :param X:
        :param t:
        :return:
        """
        f_t, F_t, S_t = self.forward(X, t=t)
        return F_t

class DeepHit(AbstractCompetingRisksSurvivalTask):
    """
    pl.Lightning Module for DeepHit.
        :param network:             The net to use
        :param n_events:            `int`   number of competing events to consider.
        :param batch_size:
        :param output_dim:           n_discrete_times that the model has. Also the size of outlayer needs to be n_events*n_discrete_times.
        :param alpha:               `float` adjustment param for the ranking and nll loss
        :param sigma:               `int`   param for the ranking loss. the higher sigma, the stronger the ranking.
        :param num_workers:     `int` nr of workers for the dataloader
        :param optimizer:       `torch.optim` class, the optimizer to apply
        :param optimizer_kwargs:    `dict` kwargs for optimizer
        :param schedule:        `LRschedule` class to use, optional
        :param schedule_kwargs: `dict` kwargs for scheduler
    """
    def __init__(self, network,
                 transforms=None,
                 n_events=1,
                 batch_size=128,
                 output_dim=20,
                 alpha=0.75,
                 sigma=3,
                 num_workers=8,
                 lr=1e-2,
                 evaluation_time_points=[10],
                 evaluation_quantile_bins=None,
                 report_train_metrics=True,
                 optimizer=torch.optim.SGD,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={},
                 **kwargs
                 ):
        super().__init__(
            network=network,
            transforms=transforms,
            n_events=n_events,
            num_workers=num_workers,
            batch_size=batch_size,
            lr=lr,
            evaluation_time_points=evaluation_time_points,
            evaluation_quantile_bins=evaluation_quantile_bins,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            schedule=schedule,
            schedule_kwargs=schedule_kwargs)

        self.n_events = n_events
        self.sigma = sigma
        self.alpha = alpha
        self.output_dim = output_dim
        self.cuts = None

        # save the params.
        self.save_hyperparameters()

    def unpack_batch(self, batch):
        data, rank_mat, (durations, events) = batch
        return data, rank_mat, durations, events

    def get_time_cuts(self, max_time=None):
        """
        Get the interval borders for the discrete times.
        :param n_durations:
        :param ds:
        :return:
        """
        if self.cuts is not None:
            return self.cuts

        if max_time is None:
            max_time = -np.inf
            for batch in self.train_dataloader():
                data, rank_mat, (durations, events) = batch
                max_duration = float(durations.max())
                if max_time < max_duration:
                    max_time = max_duration

        return np.linspace(0, max_time, self.output_dim + 1)

    def get_duration_idxs(self, durations):
        idx_durations = torch.empty(durations.size(), device=self.device).fill_(self.output_dim - 1)
        idxs = torch.Tensor(np.arange(1, self.output_dim - 1)).to(self.device)
        for i in reversed(range(1, self.output_dim - 1)):
            idx_durations = torch.where(durations < self.cuts[i], idxs[i-1], idx_durations)
        return idx_durations

    def shared_step(self, data, durations, events):
        pdf_logits = self.net(data)
        return durations, events, pdf_logits

    def training_step(self, batch, batch_idx):
        data, rank_mat, durations, events = self.unpack_batch(batch)
        if self.transforms is not None:
            data = self.transforms.apply_train_transform(data)
        args = self.shared_step(data, durations, events)
        loss_dict = self.loss(*args, rank_mat)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        data, rank_mat, durations, events = self.unpack_batch(batch)

        if self.transforms is not None:
            data = self.transforms.apply_valid_transform(data)
        args = self.shared_step(data, durations, events)
        loss_dict = self.loss(*args, rank_mat)
        return dict([(f'val_{k}', v) for k, v in loss_dict.items()])

    def loss(self, durations, events, pdf_logits, rank_mat):
        if self.cuts is None:
            self.cuts = self.get_time_cuts()
        idx_durations = self.get_duration_idxs(durations)

        pdf_logits = pdf_logits.unsqueeze(-1).view(durations.size(0),
                                                   self.n_events,
                                                   self.output_dim)

        # calculate L_1 => the nll
        nll = nll_pmf_cr(pdf_logits, idx_durations.long(), events.long(), reduction='mean')
        # calculate L_2 => the ranking loss
        rank_loss = rank_loss_deephit_cr(pdf_logits, idx_durations.long(),
                                         events.long(), rank_mat, self.sigma, reduction='mean')

        loss_val = self.alpha * nll \
                   + (1.-self.alpha) * rank_loss

        return {"nll_loss": nll,
                "rank_loss": rank_loss,
                "loss": loss_val}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["cuts"] = self.cuts

    def on_load_checkpoint(self, checkpoint):
        self.cuts = checkpoint["cuts"]

    @auto_move_data
    def predict_dataset(self, dataset, times: float):
        """
        :param dataset:
        :param times: = np array containing times at which S(t|X) shoudl be computed
        :return:
        """
        # we assume that sample is (datamodules, (duration, event))
        events = []
        durations = []
        F_t_outdim = []
        f_t_outdim = []
        t_X = []

        loader = self.ext_dataloader(dataset, batch_size=256, num_workers=4, shuffle=False, drop_last=False)
        self.cuts = self.get_time_cuts()
        self.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                data, _, d, e = self.unpack_batch(batch)

                if self.transforms is not None:
                    data = self.transforms.apply_valid_transform(data)

                durations.append(d)
                events.append(e)

                with torch.no_grad():
                    pdf_logits = self.run_net(data)  # [1024, outdim*n_events]
                    t_X.append(self.predict_event_time(data)) # [1024, n_e]
                shape = pdf_logits.size()
                pdf_logits = pdf_logits.unsqueeze(-1).view(shape[0],
                                                           self.n_events,
                                                           self.output_dim)
                f_t = pad_col(pdf_logits.view(shape[0], -1)).softmax(1)[:, :-1].view(
                    pdf_logits.size())  # [batchsize, n_events, durations]
                F_t = torch.cumsum(f_t, dim=-1)

                f_t_outdim.append(f_t.permute(1, 2, 0))  #[n_events, outdim, 1024]
                F_t_outdim.append(F_t.permute(1, 2, 0))  #[n_events, outdim, 1024]
                del data
                del pdf_logits

        f_t_outdims = torch.cat(f_t_outdim, dim=-1).detach().cpu().numpy()
        F_t_outdims = torch.cat(F_t_outdim, dim=-1).detach().cpu().numpy()
        t_Xs = np.concatenate(t_X, axis=0).transpose(1, 0)

        # DO THIS FOR EVERY EVENT:
        F_tXs = []
        f_tXs = []
        for e in range(self.n_events):
            interpolated_times = [0,]
            for i in range(1, self.output_dim):
                interpolated_times.extend(np.linspace(self.cuts[i-1], self.cuts[i], 25).tolist()[1:])
            interpolated_times.extend(np.linspace(self.cuts[-2], np.max(times), 25).tolist()[1:])

            # get pandas_dataframe
            F_t_outdim = pd.DataFrame(F_t_outdims[e], index=self.cuts[1:])
            F_t_outdim.reindex(interpolated_times).interpolate(method='linear', axis=0,
                                                               inplace=True, limit_direction='forward')
            f_t_outdim = pd.DataFrame(f_t_outdims[e], index=self.cuts[1:])
            f_t_outdim.reindex(interpolated_times).interpolate(method='linear', axis=0,
                                                               inplace=True, limit_direction='forward')
            F_tX = []
            f_tX = []
            for t in times.ravel():
                F_tX.append(F_t_outdim.iloc[[F_t_outdim.index.get_loc(t, method="nearest")]].values)
                f_tX.append(f_t_outdim.iloc[[f_t_outdim.index.get_loc(t, method="nearest")]].values)
            f_tXs.append(np.concatenate(f_tX, axis=0))
            F_tXs.append(np.concatenate(F_tX, axis=0))
        F_tXs = np.stack(F_tXs, axis=0) # [e, t, n_samples]
        f_tXs = np.stack(f_tXs, axis=0)
        S_tXs = 1 - F_tXs

        pred_df = []
        for e in range(self.n_events):
            for t_i, t in enumerate(times):
                df=pd.DataFrame.from_dict({
                    f"{e}_{t}_ft": f_tXs[e, t_i, :].ravel(),
                    f"{e}_{t}_Ft": F_tXs[e, t_i, :].ravel(),
                    f"{e}_{t}_St": S_tXs[e, t_i, :].ravel()})
                pred_df.append(df)
            pred_df[-1][f"{e}_time"] = t_Xs[e, :].ravel()
        pred_df = pd.concat(pred_df, axis=1)
        pred_df['durations'] = torch.cat(durations, dim=0).cpu().numpy().ravel()
        pred_df['events'] = torch.cat(events, dim=0).cpu().numpy().ravel()
        self.train()
        return pred_df

    def predict_event_time(self, X):
        """
        Predict the event time from datamodules.
        :param X:
        :return:
        """
        self.cuts = self.get_time_cuts()
        pdf_logits = self.run_net(X)
        f_t = pad_col(pdf_logits.view(pdf_logits.size(0), -1)).softmax(1)[:, :-1].view(
            pdf_logits.size(0), self.n_events, self.output_dim)
        max = torch.argmax(f_t, dim=-1) #[B, e]
        pred_times = []
        for e in range(self.n_events):
            pred_times.append(self.cuts[max[:, e].cpu().numpy()])
        return np.stack(pred_times, axis=-1)

    @auto_move_data
    def run_net(self, X):
        pdf_logits = self.net(X)  # [1, outdim]
        return pdf_logits

    @auto_move_data
    def forward(self, X, t=None):
        """
        Predict a sample
        :return:
        """
        idx_durations = self.get_duration_idxs(t)
        pdf_logits = self.net(X)  # [1, outdim]

        # reshape:
        f_t = pad_col(pdf_logits.view(pdf_logits.size(0), -1)).softmax(1)[:, :-1].view(
            pdf_logits.size(0), self.n_events, self.output_dim) # [B, e, outdim]

        F_t = torch.cumsum(f_t, dim=-1)[:, :, idx_durations.long()].permute(1,0,2)
        f_t = f_t[:, :, idx_durations.long()].permute(1,0,2)  # [e, B, outdim]

        return f_t, F_t, 1-F_t

    def predict_risk(self, X, t=None):
        """
        Predict Risk (= F(t)) nothing else.
        :param X:
        :param t:
        :return:
        """
        f_t, F_t, S_t = self.forward(X, t=t)
        return F_t

    @property
    def __name__(self):
        return 'DeepHit'


class BayesianDeepSurvivalMachine(DeepSurvivalMachine, AbstractBayesianSurvivalTask):
    """
    DSMs with BayesNNs.
    """
    def __init__(self,
                 network,
                 transforms=None,
                 n_events=1,
                 batch_size=128,
                 alpha=0.75,
                 gamma=1e-8,
                 num_mc=10,
                 complexity_cost_weight=1e-6,
                 k_dim=8,
                 output_dim=100,
                 temperature=100,
                 distribution='weibull',
                 num_workers=8,
                 lr=1e-3,
                 report_train_metrics=True,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={},
                 bayeslayer_kwargs={"prior_sigma_1": 0.1,
                                    "prior_sigma_2": 0.4,
                                    "prior_pi": 1,
                                    "posterior_mu_init": 0,
                                    "posterior_rho_init": -3.0
                                    },
                 **kwargs
                 ):
        """
        pl.Lightning Module for DeepSurvivalMachines.

        :param network:         `nn.Module` or `pl.LightningModule`, the network
        :param transforms:      `nn.Module` holding the transformations to apply to datamodules if necessary
        :param n_events:        `int`, number of events to consider. Minimum 1.
        :param batch_size:      `int`, batchsize
        :param alpha:           `float`, ]0, 1] weight for the loss function (1 = equal ratio, <1, upweighting of f_t(X))
        :param gamma:           `float`, factor to add to the shape and scale params to avoid edge conditions.
        :param k_dim:           `int`, number of distributions in the mixture
        :param output_dim:      `int`, outdim of `network` and in_dim to the layers generating k and `
        :param temperature:     `int`, temperature of the softmax parameter
        :param distribution:    `str` [`weibull`, `lognormal`] the base distribution
        :param num_workers:     `int` nr of workers for the dataloader
        :param optimizer:       `torch.optim` class, the optimizer to apply
        :param optimizer_kwargs:    `dict` kwargs for optimizer
        :param schedule:        `LRschedule` class to use, optional
        :param schedule_kwargs: `dict` kwargs for scheduler
        """
        super().__init__(
            network,
            transforms=transforms,
            n_events=n_events,
            batch_size=batch_size,
            alpha=alpha,
            gamma=gamma,
            k_dim=k_dim,
            output_dim=output_dim,
            temperature=temperature,
            distribution=distribution,
            num_workers=num_workers,
            lr=lr,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            schedule=schedule,
            schedule_kwargs=schedule_kwargs,
        )
        self.bayeslayer_kwargs = bayeslayer_kwargs
        self.num_mc = num_mc
        self.complexity_cost_weight = complexity_cost_weight

        self.temperature = temperature
        self.output_dim = output_dim
        self.alpha = alpha
        self.gamma = gamma
        assert distribution in ['weibull', 'lognormal'], 'Currently only `lognormal` & `weibull` available.'
        self.distribution = distribution
        # build the nets:
        self.scale = add_decorator(nn.Sequential(blitz.modules.BayesianLinear(self.output_dim, self.hparams.k_dim * self.n_events, bias=True,
                                                                              **self.bayeslayer_kwargs),
                                                 nn.Softplus(),
                                                 )).to(self.device)
        self.shape = add_decorator(nn.Sequential(blitz.modules.BayesianLinear(self.output_dim, self.hparams.k_dim * self.n_events, bias=True,
                                                                              **self.bayeslayer_kwargs),
                                                 nn.Softplus()
                                                 )).to(self.device)
        self.ks = add_decorator(blitz.modules.BayesianLinear(self.output_dim, self.hparams.k_dim * self.n_events, bias=True,
                                                             **self.bayeslayer_kwargs))

        # set params to be optimized:
        self.params = []
        self.networks = [self.net, self.scale, self.shape, self.ks]

        for n in self.networks:
            self.params.extend(list(n.parameters()))

        # save the params.
        self.save_hyperparameters()

    @property
    def __name__(self):
        return 'BayesianDeepSurvivalMachine'

    @auto_move_data
    def sample_forward(self, X, n_samples=20, forward_kwargs={}):
        """
        Sample Preds.
        :param n_samples: `int`, n_samples
        :return:
        """
        preds = [self.forward(X, **forward_kwargs) for _ in range(n_samples)]

        preds_ft = torch.stack([p[0] for p in preds], dim=0) # [n_samples, e, B]
        preds_Ft = torch.stack([p[1] for p in preds], dim=0)
        preds_St = torch.stack([p[2] for p in preds], dim=0)

        std_ft = preds_ft.std(dim=0)
        std_Ft = preds_Ft.std(dim=0)
        std_St = preds_St.std(dim=0)

        preds_ft = preds_ft.mean(dim=0)
        preds_Ft = preds_Ft.mean(dim=0)
        preds_St = preds_St.mean(dim=0)

        return (preds_ft, std_ft), (preds_Ft, std_Ft), (preds_St, std_St)

    @auto_move_data
    def predict_risk(self, X, t=None):
        """
        We predict the median quantile for the estimated event time.
        :param X:
        :return:
        """
        _, (preds, _),_ = self.sample_forward(X, n_samples=10, forward_kwargs={"t": t}) # Ft
        return preds

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
        f_tX, F_tX, S_tX, t_X = [], [], [], []
        f_tX_std, F_tX_std, S_tX_std, t_X_std = [], [], [], []

        with torch.no_grad():
            for batch in loader:
                data, d, e = self.unpack_batch(batch)
                if self.transforms is not None:
                    data = self.transforms.apply_valid_transform(data)
                durations.append(d)
                events.append(e)
                f_sample, F_sample, S_sample = [], [], []
                f_sample_std, F_sample_std, S_sample_std = [], [], []
                for t in times:
                    f_preds, F_preds, S_preds = self.sample_forward(data, n_samples=20, forward_kwargs={'t': torch.Tensor([t])})
                    f_sample.append(f_preds[0])
                    F_sample.append(F_preds[0])
                    S_sample.append(S_preds[0])
                    f_sample_std.append(f_preds[1])
                    F_sample_std.append(F_preds[1])
                    S_sample_std.append(S_preds[1])

                samples = [self.sample_eventtime_from_mixture(data) for _ in range(25)]
                t_sample = torch.stack([s[0] for s in samples], dim=0)  # [n_samples, e, B]
                t_sample = t_sample.mean(dim=0)
                # t_sample_std = [s[1] for s in samples]

                del data
                t_X.append(t_sample.permute(1, 0))
                # t_X_std.append(t_sample_std.permute(1, 0))
                S_tX.append(torch.stack(S_sample, dim=1))
                F_tX.append(torch.stack(F_sample, dim=1))
                f_tX.append(torch.stack(f_sample, dim=1))
                S_tX_std.append(torch.stack(S_sample_std, dim=1))
                F_tX_std.append(torch.stack(F_sample_std, dim=1))
                f_tX_std.append(torch.stack(f_sample_std, dim=1))
        t_X = torch.cat(t_X, dim=-1).detach().cpu().numpy()  # -> [e, t, n_samples]
        # t_X_std = torch.cat(t_X_std, dim=-1).detach().cpu().numpy()  # -> [e, t, n_samples]
        S_tX = torch.cat(S_tX, dim=-1).detach().cpu().numpy()  # -> [e, t, n_samples]
        F_tX = torch.cat(F_tX, dim=-1).detach().cpu().numpy()
        f_tX = torch.cat(f_tX, dim=-1).detach().cpu().numpy()
        S_tX_std = torch.cat(S_tX_std, dim=-1).detach().cpu().numpy()  # -> [e, t, n_samples]
        F_tX_std = torch.cat(F_tX_std, dim=-1).detach().cpu().numpy()
        f_tX_std = torch.cat(f_tX_std, dim=-1).detach().cpu().numpy()

        self.train()
        pred_df = []
        for e in range(self.n_events):
            for t_i, t in enumerate(times):
                df=pd.DataFrame.from_dict({
                    f"{e}_{t}_ft": f_tX[e, t_i, :].ravel(),
                    f"{e}_{t}_ft_stds": f_tX_std[e, t_i, :].ravel(),
                    f"{e}_{t}_Ft": F_tX[e, t_i, :].ravel(),
                    f"{e}_{t}_Ft_stds": F_tX_std[e, t_i, :].ravel(),
                    f"{e}_{t}_St": S_tX[e, t_i, :].ravel(),
                    f"{e}_{t}_St_stds": S_tX_std[e, t_i, :].ravel()})
                pred_df.append(df)
            pred_df[-1][f'{e}_time'] = t_X[e, :].ravel()
            # pred_df[-1][f'{e}_time_std'] = t_X_std[e, :].ravel()
        pred_df = pd.concat(pred_df, axis=1)
        pred_df['durations'] = torch.cat(durations, dim=0).cpu().numpy().ravel()
        pred_df['events'] = torch.cat(events, dim=0).cpu().numpy().ravel()
        return pred_df

    def training_step(self, batch, batch_idx):
        return AbstractBayesianSurvivalTask.training_step(self, batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return AbstractBayesianSurvivalTask.validation_step(self, batch, batch_idx)


class BayesianDeepHit(DeepHit, AbstractBayesianSurvivalTask):
    """
    pl.Lightning Module for DeepHit.
        :param network:             The net to use
        :param n_events:            `int`   number of competing events to consider.
        :param batch_size:
        :param output_dim:           n_discrete_times that the model has. Also the size of outlayer needs to be n_events*n_discrete_times.
        :param alpha:               `float` adjustment param for the ranking and nll loss
        :param sigma:               `int`   param for the ranking loss. the higher sigma, the stronger the ranking.
        :param num_workers:     `int` nr of workers for the dataloader
        :param optimizer:       `torch.optim` class, the optimizer to apply
        :param optimizer_kwargs:    `dict` kwargs for optimizer
        :param schedule:        `LRschedule` class to use, optional
        :param schedule_kwargs: `dict` kwargs for scheduler
    """
    def __init__(self,
                 network,
                 transforms=None,
                 n_events=1,
                 batch_size=128,
                 output_dim=20,
                 alpha=0.75,
                 sigma=3,
                 num_mc=5,
                 complexity_cost_weight=1e-6,
                 num_workers=8,
                 lr=1e-3,
                 report_train_metrics=True,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={},
                 **kwargs
                 ):
        super().__init__(network=network, transforms=transforms,
                         n_events=n_events, batch_size=batch_size, output_dim=output_dim,
                         alpha=alpha, sigma=sigma, num_workers=num_workers, lr=lr,
                         optimizer=optimizer, optimizer_kwargs=optimizer_kwargs, schedule=schedule,
                         schedule_kwargs=schedule_kwargs)

        self.num_mc = num_mc
        self.complexity_cost_weight = complexity_cost_weight

        # save the params.
        self.save_hyperparameters()

    @property
    def __name__(self):
        return 'BayesianDeepHit'

    def predict_risk(self, X, t=None):
        """
        Predict Risk (= F(t)) nothing else.
        :param X:
        :param t:
        :return:
        """
        f_t, F_t, S_t = self.sample_forward(X, forward_kwargs={'t':t}, n_samples=10)
        return F_t[0]

    @auto_move_data
    def sample_forward(self, X, n_samples=20, forward_kwargs={}):
        """
        Sample Preds.
        :param n_samples: `int`, n_samples
        :return:
        """
        preds = [self.forward(X, **forward_kwargs) for _ in range(n_samples)] # [e, B, t]

        preds_ft = torch.stack([p[0] for p in preds], dim=0) # [n_samples, e, B, 1]
        preds_Ft = torch.stack([p[1] for p in preds], dim=0)
        preds_St = torch.stack([p[2] for p in preds], dim=0)

        std_ft = preds_ft.std(dim=0)
        std_Ft = preds_Ft.std(dim=0)
        std_St = preds_St.std(dim=0)

        preds_ft = preds_ft.mean(dim=0)
        preds_Ft = preds_Ft.mean(dim=0)
        preds_St = preds_St.mean(dim=0)

        return (preds_ft, std_ft), (preds_Ft, std_Ft), (preds_St, std_St)

    def training_step(self, batch, batch_idx):
        data, rank_mat, durations, events = self.unpack_batch(batch)
        if self.transforms is not None:
            data = self.transforms.apply_train_transform(data, events)
        loss = self.sample_elbo(data, durations, events, rank_mat)
        self.log(f"train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def sample_elbo(self, data, durations, events, rank_mat):
        """
        Here we need to sample the elbo over num_mc passes.
        :return:
        """
        loss = 0
        for _ in range(self.num_mc):
            args = self.shared_step(data, durations, events)

            loss += self.loss(*args, rank_mat)['loss']
            for net in self.networks:
                try:
                    loss += net.nn_kl_divergence() * self.complexity_cost_weight
                except AttributeError:
                    continue
        return loss / self.num_mc

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
        f_tX, F_tX, S_tX = [], [], []
        t_X, t_X_std = [], []
        f_tX_std, F_tX_std, S_tX_std = [], [], []

        with torch.no_grad():
            for batch in loader:
                data, _, d, e = self.unpack_batch(batch)
                if self.transforms is not None:
                    data = self.transforms.apply_valid_transform(data)
                durations.append(d)
                events.append(e)
                f_sample, F_sample, S_sample = [], [], []
                f_sample_std, F_sample_std, S_sample_std = [], [], []
                for t in times:
                    f_preds, F_preds, S_preds = self.sample_forward(data, n_samples=20, forward_kwargs={'t': torch.Tensor([t])})
                    f_sample.append(f_preds[0])
                    F_sample.append(F_preds[0])
                    S_sample.append(S_preds[0])
                    f_sample_std.append(f_preds[1])
                    F_sample_std.append(F_preds[1])
                    S_sample_std.append(S_preds[1])

                pred_times = np.stack([self.predict_event_time(data) for i in range(20)], axis=-1)
                del data
                t_X.append(pred_times.mean(axis=-1)) #[B, e]
                t_X_std.append(pred_times.std(axis=-1))
                S_tX.append(torch.cat(S_sample, dim=-1))
                F_tX.append(torch.cat(F_sample, dim=-1))
                f_tX.append(torch.cat(f_sample, dim=-1))
                S_tX_std.append(torch.cat(S_sample_std, dim=-1))
                F_tX_std.append(torch.cat(F_sample_std, dim=-1))
                f_tX_std.append(torch.cat(f_sample_std, dim=-1))
        S_tX = torch.cat(S_tX, dim=1).detach().permute(0,2,1).cpu().numpy()  # -> [e, n_samples, t]
        F_tX = torch.cat(F_tX, dim=1).detach().permute(0,2,1).cpu().numpy()
        f_tX = torch.cat(f_tX, dim=1).detach().permute(0,2,1).cpu().numpy()
        S_tX_std = torch.cat(S_tX_std, dim=1).detach().permute(0,2,1).cpu().numpy()
        F_tX_std = torch.cat(F_tX_std, dim=1).detach().permute(0,2,1).cpu().numpy()
        f_tX_std = torch.cat(f_tX_std, dim=1).detach().permute(0,2,1).cpu().numpy()
        t_X = np.concatenate(t_X, axis=0).transpose(1, 0)  # [n_samples, e] -> [e, n_samples]
        t_X_std = np.concatenate(t_X_std, axis=0).transpose(1, 0)  # [n_samples, e] -> [e, n_samples]

        self.train()
        pred_df = []
        for e in range(self.n_events):
            for t_i, t in enumerate(times):
                df = pd.DataFrame.from_dict({
                    f"{e}_{t}_ft": f_tX[e, t_i, :].ravel(),
                    f"{e}_{t}_ft_stds": f_tX_std[e, t_i, :].ravel(),
                    f"{e}_{t}_Ft": F_tX[e, t_i, :].ravel(),
                    f"{e}_{t}_Ft_stds": F_tX_std[e, t_i, :].ravel(),
                    f"{e}_{t}_St": S_tX[e, t_i, :].ravel(),
                    f"{e}_{t}_St_stds": S_tX_std[e, t_i, :].ravel()})
                pred_df.append(df)
            pred_df[-1][f"{e}_time"] = t_X[e, :].ravel()
            pred_df[-1][f"{e}_time_std"] = t_X_std[e, :].ravel()
        pred_df = pd.concat(pred_df, axis=1)
        pred_df['durations'] = torch.cat(durations, dim=0).cpu().numpy().ravel()
        pred_df['events'] = torch.cat(events, dim=0).cpu().numpy().ravel()
        return pred_df

    def validation_step(self, batch, batch_idx):
        return DeepHit.validation_step(self, batch, batch_idx)


class GraphDeepSurvivalMachine(DeepSurvivalMachine):
    """
    DeepSurvivalMachine with graph nn as feature extractor.
    """
    def __init__(self,
                 network=None,
                 etypes=[],
                 ntypes=[],
                 transforms=None,
                 n_events=1,
                 event_names=None,
                 batch_size=1024,
                 alpha=1,
                 gamma=1e-8,
                 k_dim=8,
                 output_dim=100,
                 temperature=1000,
                 distribution='weibull',
                 num_workers=8,
                 lr=1e-2,
                 report_train_metrics=True,
                 optimizer=torch.optim.SGD,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={},
                 **kwargs
                 ):
        super().__init__(
            network,
            transforms=transforms,
            n_events=n_events,
            event_names=event_names,
            batch_size=batch_size,
            alpha=alpha,
            gamma=gamma,
            k_dim=k_dim,
            output_dim=output_dim,
            temperature=temperature,
            distribution=distribution,
            num_workers=num_workers,
            lr=lr,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            schedule=schedule,
            schedule_kwargs=schedule_kwargs
        )
        self.etypes = etypes
        self.ntypes = ntypes
        self.surv_train = None
        self.surv_valid = None

    def unpack_batch(self, batch):
        input_nodes, output_nodes, blocks = batch
        # get embedding here?
        inputs = blocks[0].srcdata['features']
        durations = blocks[-1].nodes['Patient'].data['duration']  # returns a dict
        events = blocks[-1].nodes['Patient'].data['event']  # returns a dict
        return (blocks, inputs), durations, events

    def validation_epoch_end(self, outputs):
        metrics = {}
        # aggregate the per-batch-metrics:
        for metric_name in ["val_loss"]:
            # for metric_name in [k for k in outputs[0].keys() if k.startswith("val")]:
            metrics[metric_name] = torch.stack([output[metric_name] for output in outputs]).mean()

        # calculate the survival metrics
        metrics_survival = self.calculate_survival_metrics()

        metrics.update(metrics_survival)

        for key, value in metrics.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def prepare_survival_metrics(self):
        """
        Estimate quantile bins for Ctd.
        :param quantile_bins:
        :return:
        """
        valid_loader = self.val_dataloader()
        train_loader = self.train_dataloader()

        self.surv_train = np.stack([train_loader.events.values, train_loader.durations.values], axis=0).squeeze(axis=-1)
        self.surv_valid = np.stack([valid_loader.events.values, valid_loader.durations.values], axis=0).squeeze(axis=-1)


    def calculate_survival_metrics(self, train_ds, valid_ds, time_points=[5, 10], quantile_bins=None):
        """
        THIS IS THE COMPETING EVENTS VERSION!

        1. Calculate the Ctd on the quartiles of the valid set.
        2. Calculate the Brier scires for the same times.
        :return:
        """
        metrics = {}

        assert None in [time_points, quantile_bins], 'EITHER pass quantiles OR pass timepoints'

        try:
            surv_train = np.stack([train_ds.events.values, train_ds.durations.values], axis=0).squeeze(axis=-1)
            surv_valid = np.stack([valid_ds.events.values, valid_ds.durations.values], axis=0).squeeze(axis=-1)
        except AttributeError:
            surv_train = np.stack([train_ds.events, train_ds.durations], axis=0).squeeze(axis=-1)
            surv_valid = np.stack([valid_ds.events, valid_ds.durations], axis=0).squeeze(axis=-1)

        self.eval()

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
                for batch in self.val_dataloader():
                    data, d, e = self.unpack_batch(batch)

                    if self.transforms is not None:
                        data = self.transforms.apply_valid_transform(data)
                    risk = self.predict_risk(data, t=tau_)
                    del data
                    risks.append(risk.detach().cpu().numpy())

            risks = np.concatenate(risks, axis=1)

            c_per_event = []
            ctd_per_event = []
            for e in range(1, self.n_events + 1):
                e_risks = risks[e-1].ravel()
                e_risks[pd.isna(e_risks)] = np.nanmax(e_risks)
                # move to structured arrays:
                struc_surv_train = np.array([(1 if e_ == e else 0, d) for e_, d in zip(surv_train[0], surv_train[1])],
                                            dtype=[('event', 'bool'), ('duration', 'f8')])
                struc_surv_valid = np.array([(1 if e_ == e else 0, d) for e_, d in zip(surv_valid[0], surv_valid[1])],
                                            dtype=[('event', 'bool'), ('duration', 'f8')])

                Ctd = concordance_index_ipcw(struc_surv_train, struc_surv_valid,
                                             e_risks,
                                             tau=tau, tied_tol=1e-8)
                C = concordance_index(event_times=surv_valid[1],
                                      predicted_scores=-risks,
                                      event_observed=surv_valid[0])
                ctd_per_event.append(Ctd[0])
                c_per_event.append(C)
                metrics[f'{self.event_names[e-1]}_Ctd_{annot[i]}'] = Ctd[0]
                metrics[f'{self.event_names[e-1]}_C_{annot[i]}'] = C

            metrics[f'Ctd_{annot[i]}'] = np.asarray(ctd_per_event).mean()
            metrics[f'C_{annot[i]}'] = np.asarray(c_per_event).mean()

        self.train()
        return metrics


class CSurv(AbstractCompetingRisksSurvivalTask):
    def __init__(self,
                 network,
                 transforms=None,
                 n_events=1,
                 batch_size=128,
                 alpha=0.5,
                 num_workers=8,
                 lr=1e-2,
                 gamma=1e-8,
                 tmax=25,
                 aug_factor=1,
                 diff=False,
                 epsilon=1e-8,
                 evaluation_time_points=[10],
                 evaluation_quantile_bins=None,
                 report_train_metrics=True,
                 optimizer=torch.optim.SGD,
                 optimizer_kwargs={},
                 schedule=None,
                 schedule_kwargs={},
                 **kwargs
                 ):
        super().__init__(
            network=network,
            transforms=transforms,
            n_events=n_events,
            num_workers=num_workers,
            batch_size=batch_size,
            lr=lr,
            evaluation_time_points=evaluation_time_points,
            evaluation_quantile_bins=evaluation_quantile_bins,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            schedule=schedule,
            schedule_kwargs=schedule_kwargs)

        self.n_events = n_events
        self.alpha = alpha
        self.gamma = gamma
        self.aug_factor = aug_factor
        self.diff = diff
        self.epsilon = epsilon
        self.tmax = tmax

        # save the params.
        self.save_hyperparameters()

    def _normalize_timepoints(self, t):
        return (t-1).div((self.tmax-1))

    def unpack_batch(self, batch):
        data, (durations, events) = batch
        return data, durations, events

    def shared_step(self, data, durations, events):
        scaled_durations = self._normalize_timepoints(torch.clone(durations))
        if self.diff:
            with torch.enable_grad():
                times = torch.autograd.Variable(scaled_durations, requires_grad=True)
                F_ts = self.net(data, times)
                f_ts, = torch.autograd.grad(
                    outputs=[F_ts],
                    inputs=[times],
                    grad_outputs=torch.ones_like(F_ts),
                    retain_graph=True,
                    create_graph=True,
                    only_inputs=True,
                    allow_unused=True
                )
        else:
            F_ts = self.net(data, scaled_durations)
            F_ts_epsilon = self.net(data, scaled_durations+self.epsilon)

            f_ts = (F_ts_epsilon - F_ts)/self.epsilon

        return durations, events, F_ts, f_ts

    def augment_batch(self, data, durations, events):
        """
        Sample event times from a uniform distribution between 0 and tmax (batch). FOR THE OBSERVED EVENTS
        We sample 1 left and 1 right of the event time!

        :param data:
        :param durations:
        :param events:
        :return:
        """
        # get all samples with an event:
        obs_events = events > 0
        indices = obs_events.nonzero(as_tuple=True)[0]
        tmax = durations.max()
        n_indices = indices.size(0)

        # sample from 0 to event time:
        # (r2 - r1) * torch.rand(a, b) + r1
        lower_samples = []
        upper_samples = []
        for i in range(self.aug_factor):
            l = durations[indices].squeeze(-1) * torch.rand(n_indices).to(durations.device)
            lower_samples.append(l)
            u = (tmax.repeat(indices.size(0)) - durations[indices].squeeze(-1)) * torch.rand(n_indices).to(durations.device)
            upper_samples.append(u)
        lower_samples = torch.cat(lower_samples, dim=0)
        upper_samples = torch.cat(upper_samples, dim=0)

        # augment:
        data = torch.cat([data]+ 2*self.aug_factor*[data[indices]], dim=0)
        durations = torch.cat([durations, lower_samples.unsqueeze(1), upper_samples.unsqueeze(1)], dim=0)
        events = torch.cat([events]+self.aug_factor*[torch.zeros_like(events[indices])]+self.aug_factor*[torch.ones_like(events[indices])], dim=0)

        # AUGMENT FOR CENSORED:
        obs_events = events < 1
        indices = obs_events.nonzero(as_tuple=True)[0]
        n_indices = indices.size(0)

        # sample from 0 to event time:
        # (r2 - r1) * torch.rand(a, b) + r1
        lower_samples = []
        for i in range(self.aug_factor):
            l = durations[indices].squeeze(-1) * torch.rand(n_indices).to(durations.device)
            lower_samples.append(l)
        lower_samples = torch.cat(lower_samples, dim=0)

        # augment:
        data = torch.cat([data]+ self.aug_factor*[data[indices]], dim=0)
        durations = torch.cat([durations, lower_samples.unsqueeze(1)], dim=0)
        events = torch.cat([events]+self.aug_factor*[torch.zeros_like(events[indices])], dim=0)

        return data, durations, events

    def training_step(self, batch, batch_idx):
        data, durations, events = self.unpack_batch(batch)
        if self.transforms is not None:
            data = self.transforms.apply_train_transform(data)

        # augment:
        data, durations, events = self.augment_batch(data, durations, events)
        args = self.shared_step(data, durations, events)
        loss_dict = self.loss(*args)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        data, durations, events = self.unpack_batch(batch)

        if self.transforms is not None:
            data = self.transforms.apply_valid_transform(data)

        args = self.shared_step(data, durations, events)
        loss_dict = self.loss(*args)
        return dict([(f'val_{k}', v) for k, v in loss_dict.items()])

    def loss(self, durations, events, F_ts, f_ts):
        loss_dict = {}
        for e in range(1, self.n_events+1):
            e_loss_dict = {f"loss_{e}": 0.,
                      f"uc_loss_Ft_{e}": 0.,
                      f"uc_loss_ft_{e}": 0.,
                      f"c_loss_Ft_{e}": 0.}
            e_ = torch.Tensor([e])
            e_ = e_.type_as(F_ts)
            # censored loss => maximise S(t|X) => minimize nll(S(t|X)
            mask = torch.ne(events, e_)
            c_loss_Ft = -torch.log(1 - F_ts + self.gamma)
            c_loss_Ft = c_loss_Ft[mask]
            c_loss_Ft = c_loss_Ft.sum()/mask.sum()
            e_loss_dict[f"c_loss_Ft_{e}"] = c_loss_Ft
            e_loss_dict[f"loss_{e}"] += self.alpha * c_loss_Ft

            # uncensored FAKE loss => maximise F(t|X)
            mask = torch.eq(events, e_)
            uc_loss_Ft = -torch.log(self.gamma+F_ts)
            uc_loss_Ft = uc_loss_Ft[mask]
            uc_loss_Ft = uc_loss_Ft.sum()/mask.sum()
            e_loss_dict[f"uc_loss_Ft_{e}"] = uc_loss_Ft
            e_loss_dict[f"loss_{e}"] += (1-self.alpha) * uc_loss_Ft

            # autodiff (Ft, t) => ft, calculate this loss only for real obs events
            f_ts_real = f_ts[:self.hparams.batch_size, :]
            mask = torch.eq(events[:self.hparams.batch_size, :], e_)
            f_ts_real = f_ts_real[mask]

            # uc_loss_ft = -torch.log(self.gamma+f_ts_real)
            # uc_loss_ft = uc_loss_ft.sum()/mask.sum() if not torch.eq(mask.sum(), torch.Tensor([0.]).type_as(F_ts)) else 1.
            # e_loss_dict[f"uc_loss_ft_{e}"] = uc_loss_ft
            # e_loss_dict[f"loss_{e}"] += (1-self.alpha) * uc_loss_ft if torch.all(uc_loss_ft.isfinite()) else 0.

            loss_dict.update(e_loss_dict)

        loss = 0.
        for k in [k for k in loss_dict.keys() if k.startswith('loss')]:
            loss += loss_dict[k]
        loss_dict["loss"] = loss

        return loss_dict

    @auto_move_data
    def forward(self, X, t=None):
        """
        Predict a sample
        :return:
        """
        if t is None:
            raise ValueError('This model needs a timepoint for inference.')

        if X.size(0) != t.size(0):
            t = t.repeat(X.size(0), 1)

        scaled_t = self._normalize_timepoints(t)

        if self.diff:
            with torch.enable_grad():
                times = torch.autograd.Variable(scaled_t, requires_grad=True)
                F_ts = self.net(X, times)
                f_ts, = torch.autograd.grad(
                    outputs=[F_ts],
                    inputs=[times],
                    grad_outputs=torch.ones_like(F_ts),
                    retain_graph=True,
                    create_graph=True,
                    only_inputs=True,
                    allow_unused=True
                )

        else:
            F_ts = self.net(X, scaled_t)
            F_ts_epsilon = self.net(X, scaled_t + self.epsilon)

            f_ts = (F_ts_epsilon - F_ts) / self.epsilon

        return F_ts, f_ts

    def predict_risk(self, X, t=None):
        """
        Predict Risk (= F(t)) nothing else.
        :param X:
        :param t:
        :return:
        """
        F_t, f_t = self.forward(X, t=t)

        return F_t.unsqueeze(0)


    @property
    def __name__(self):
        return 'CSurv'


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
                 transforms=None,
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
            transforms=transforms,
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
                 transforms=None,
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
            transforms=transforms,
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
