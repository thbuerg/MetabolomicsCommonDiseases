import pathlib
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from riskiano.source.datasets import TabularDataset, DatasetWrapper, BatchedDS, ExclusionMaskDataset
from omegaconf import OmegaConf, ListConfig, DictConfig


class RiskianoDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=128, num_workers=8, tabular_filepath='', use_batched_ds=False,
                 return_rank_mat=None, output_dim=None, fast_dev_run=None, cv_partition=None, **kwargs):
        """
        Abstract DataModule Class for Riskiano.

        The __init__ of this calss should be called in every inherited class.

        A few points to consider:
            - hardcode filepaths for versioning
            - make durations, events and other labels explicit attributes
            - define transformations etc in the `__init__()`
            - The logic of how exactly the individual datasets are instantiated should be defined in the
              `get_dataset()` method. This method NEEDS TO BE DEFINED PER USECASE, and will be called in `setup()`.

        :param batch_size: `int`, batchsize to use, needs to be passed for the BatchedDS
        :param num_workers: `int`, number of workers for the DataLoaders
        :param use_batched_ds: `bool`, whether to use the batchedDS (`True`) or not (`False`). Defaults to `False`.
        :param output_categorical: `bool`, whether to output categorical columns (`True`) vs. 1-hot/binary columns (`False`)
        :param return_rank_mat: `bool`, whether to return the rank_mat for DeepHitTraining
        :param output_dim: `int`, output-dimension of the network, needed for cuts and rank_mat calculation, can be ommitted of rank_mat equals False
        :param fast_dev_run: `bool`, similar to pl.Trainer FLAG. in this case limits the eid_map to 100 eids.
        :param kwargs:
        """
        super().__init__()
        self.cv_partition = cv_partition
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tabular_filepath = tabular_filepath
        self.use_batched_ds = use_batched_ds
        self.fast_dev_run = fast_dev_run

        self.return_rank_mat = return_rank_mat
        if self.return_rank_mat:
            assert output_dim is not None, 'Rank mat computation needs out_dim!'
        self.output_dim = output_dim
        self.cuts = None

    def get_batched_ds(self, ds):
        if self.return_rank_mat:
            raise NotImplementedError()
        else:
            return BatchedDS(ds, batch_size=self.batch_size)

    def get_dataset(self, split):
        raise NotImplementedError('Implement according to usecase.')

    def setup(self, stage=None):
        self.train_ds = self.get_dataset('train')
        self.valid_ds = self.get_dataset('valid')
        try:
            self.test_ds = self.get_dataset('test')
        except AssertionError:
            print('No test split defined to this data.')

        if self.return_rank_mat:
            self.cuts = self.get_time_cuts()

    def get_time_cuts(self, max_time=None):
        """
        Get the interval borders for the discrete times.
        :param n_durations:
        :param ds:
        :return:
        """
        if self.cuts is not None:
            return self.cuts

        loader = DataLoader(self.train_ds, batch_size=1024, num_workers=self.num_workers, shuffle=False, drop_last=False)

        if max_time is None:
            max_time = -np.inf
            for data in loader:
                _, (durations, _) = data
                max_duration = float(durations.max())
                if max_time < max_duration:
                    max_time = max_duration
        return np.linspace(0, max_time, self.output_dim + 1)

    def train_dataloader(self):
        if self.use_batched_ds:
            return DataLoader(self.get_batched_ds(self.train_ds),
                              num_workers=self.num_workers, pin_memory=True, collate_fn=BatchedDS.default_collate,
                              shuffle=True)
        else:
            return DataLoader(self.train_ds, batch_size=self.batch_size,
                              num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        if self.use_batched_ds:
            return DataLoader(self.get_batched_ds(self.valid_ds),
                              num_workers=self.num_workers, pin_memory=True, collate_fn=BatchedDS.default_collate,
                              shuffle=False)
        else:
            return DataLoader(self.valid_ds, batch_size=self.batch_size,
                              num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        if not self.use_batched_ds:
            return DataLoader(self.test_ds, batch_size=self.batch_size,
                              num_workers=self.num_workers, shuffle=False)
        else:
            return DataLoader(self.get_batched_ds(self.test_ds),
                              num_workers=self.num_workers, pin_memory=True, collate_fn=BatchedDS.default_collate,
                              shuffle=False)


class UKBBSurvivalDatamodule(RiskianoDataModule):
    """
    Datamodule for survival training on UKBB data.

    :param batch_size: `int`, batchsize needed for outputting the rankmat + loaders
    :param num_workers: `int`, num_workers
    :param tabular_filepath: `str`, path to the ukbb data file.
    :param use_batched_ds: `bool`, whether to use the batched_dataset.
    :param features: `Union([dict, list]), features/covariates to use.
    :param duration: `str`, the duration col in the datset file
    :param event: `str`, the event col in the datset file
    :param return_rank_mat: `bool`, whether to return rank_mat (required to DeepHit Model) or not, default=False
    :param output_dim: `int`, n-timepoints in descrete time model, required for rank_mat generation -> required for DeepHit Model
    :param fast_dev_run: `bool`, run smoke test, default=False
    :param cv_partition: `int`, partition to read data from.
    :param output_categorical: `bool`, wheter to ourput raw categories (ints -> True) or do 1-hot encoding (False), Default=False
    :param exclusion_criteria: `dict`, dict of the form {`sets_apply`: [`train`, `valid`]} -> to which sets to apply exclusion criteria. Default = None
    :param kwargs:
    """
    def __init__(self,
                 batch_size=128,
                 num_workers=8,
                 tabular_filepath="",
                 use_batched_ds=False,
                 features={},
                 duration='',
                 event='',
                 return_rank_mat=None,
                 output_dim=None,
                 clip=False,
                 fast_dev_run=False,
                 cv_partition=0,
                 output_categorical=False,
                 cohort_definition=None,
                 oversampling=False,
                 **kwargs):
        super().__init__(batch_size=batch_size, num_workers=num_workers, tabular_filepath=tabular_filepath,
                         use_batched_ds=use_batched_ds,
                         return_rank_mat=return_rank_mat, output_dim=output_dim, fast_dev_run=fast_dev_run)

        self.cv_partition=cv_partition
        self.cohort_definition = cohort_definition if not isinstance(cohort_definition, DictConfig) \
            else OmegaConf.to_container(cohort_definition, resolve=True)

        assert isinstance(features, (dict, list, ListConfig, DictConfig)), 'Features must be dict or list.'

        features = features if not isinstance(features, (ListConfig, DictConfig)) \
            else OmegaConf.to_container(features, resolve=True)

        if isinstance(features, dict):
            if output_categorical == False:
                self.features = {**features["one_hot_enc"], **features["general"]}
            else:
                self.features = {**features["categorical"], **features["general"]}
            print(self.features)
            self.features = [f for group_list in self.features.values() for f in group_list]
        else:
            self.features = features

        print(type(self.features))

        self.duration = duration if not isinstance(duration, (ListConfig, DictConfig)) \
            else OmegaConf.to_container(duration, resolve=True)
        self.event = event if not isinstance(event, (ListConfig, DictConfig)) \
            else OmegaConf.to_container(event, resolve=True)
        self.clip = clip
        self.oversampling = oversampling

    def get_dataset(self, split):
        filepath = f'{self.tabular_filepath}/partition_{self.cv_partition}/{split}/data_imputed_normalized.feather'
        print(filepath)
        if self.cohort_definition is not None:
            if split in self.cohort_definition.keys():
                    eids = pd.read_feather(f"{self.tabular_filepath}/data_merged.feather").query(self.cohort_definition[split]).eid.to_list()
            else:
                eids = None
        else:
            eids = None

        ds = TabularDataset(filepath, self.features, eid_selection_mask=eids)
        if self.clip:
            upperq = ds.eid_map.quantile(.99)
            lowerq = ds.eid_map.quantile(.01)
            for c in self.features:
                ds.eid_map.loc[:, c] = ds.eid_map[c].clip(
                    lower=lowerq[c], upper=upperq[c])
        covariate_datasets = [ds]
        label_datasets = [TabularDataset(filepath, self.duration, eid_selection_mask=eids),
                          TabularDataset(filepath, self.event, eid_selection_mask=eids)]

        # make sure we have observations for each label:
        print(split)
        print(label_datasets[1].eid_map[[c for c in label_datasets[1].eid_map.columns if 'event' in c]].sum())

        # oversample if needed:
        if split == 'train' and self.oversampling:
            assert len(self.event) == 1, 'Oversampling only possible for single events.'
            pos_eids = label_datasets[1].eid_map.query(f'{self.event[0]}==1').index.values
            # augment sets:
            for ds_list in [covariate_datasets, label_datasets]:
                for ds in ds_list:
                    pos_ds = pd.concat(10*[ds.eid_map.loc[pos_eids].copy()], axis=0)
                    print(pos_ds.head())
                    pos_ds = pos_ds.reset_index(drop=True)
                    pos_ds.index.name = 'eid'
                    print(pos_ds.head())
                    ds.eid_map = pd.concat([ds.eid_map, pos_ds], axis=0)

            # make sure we have observations for each label:
            print(split)
            print(label_datasets[1].eid_map[[c for c in label_datasets[1].eid_map.columns if 'event' in c]].sum())

        return DatasetWrapper(covariate_datasets, label_datasets)


class UKBBSurvivalDatamoduleWithExclusions(UKBBSurvivalDatamodule):
    """
    Datamodule for survival training on UKBB data, that explicitly generates exclusion masks for the model.

    :param batch_size: `int`, batchsize needed for outputting the rankmat + loaders
    :param num_workers: `int`, num_workers
    :param tabular_filepath: `str`, path to the ukbb data file.
    :param use_batched_ds: `bool`, whether to use the batched_dataset.
    :param features: `Union([dict, list]), features/covariates to use.
    :param duration: `str`, the duration col in the datset file
    :param event: `str`, the event col in the datset file
    :param return_rank_mat: `bool`, whether to return rank_mat (required to DeepHit Model) or not, default=False
    :param output_dim: `int`, n-timepoints in descrete time model, required for rank_mat generation -> required for DeepHit Model
    :param fast_dev_run: `bool`, run smoke test, default=False
    :param cv_partition: `int`, partition to read data from.
    :param output_categorical: `bool`, wheter to ourput raw categories (ints -> True) or do 1-hot encoding (False), Default=False
    :param exclusion_criteria: `dict`, dict of the form {`sets_apply`: [`train`, `valid`]} -> to which sets to apply exclusion criteria. Default = None
    :param kwargs:
    """
    def __init__(self,
                 batch_size=128,
                 num_workers=8,
                 tabular_filepath="",
                 use_batched_ds=False,
                 features={},
                 duration='',
                 event='',
                 return_rank_mat=None,
                 output_dim=None,
                 clip=False,
                 fast_dev_run=False,
                 cv_partition=0,
                 output_categorical=False,
                 cohort_definition=None,
                 oversampling=False,
                 **kwargs):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            tabular_filepath=tabular_filepath,
            use_batched_ds=use_batched_ds,
            features=features,
            duration=duration,
            event=event,
            return_rank_mat=return_rank_mat,
            output_dim=output_dim,
            clip=clip,
            fast_dev_run=fast_dev_run,
            cv_partition=cv_partition,
            output_categorical=output_categorical,
            cohort_definition=None,
            oversampling=oversampling)

        # self.cohort_definition = cohort_definition if not isinstance(cohort_definition, DictConfig) \
        #     else OmegaConf.to_container(cohort_definition, resolve=True)
        self.cohort_definition = cohort_definition

    def get_dataset(self, split):
        filepath = f'{self.tabular_filepath}/partition_{self.cv_partition}/{split}/data_imputed_normalized.feather'
        print(filepath)
        if self.cohort_definition is not None:
            if split in self.cohort_definition.general.keys():
                eids = pd.read_feather(f"{self.tabular_filepath}/data_merged.feather").query(self.cohort_definition.general[split]).eid.to_list()
            else:
                eids = None
        else:
            eids = None

        ds = TabularDataset(filepath, self.features, eid_selection_mask=eids)
        if self.clip:
            upperq = ds.eid_map.quantile(.99)
            lowerq = ds.eid_map.quantile(.01)
            for c in self.features:
                ds.eid_map.loc[:, c] = ds.eid_map[c].clip(
                    lower=lowerq[c], upper=upperq[c])
        mask_ds = ExclusionMaskDataset(filepath, exclusion_criteria_dict=self.cohort_definition.task_specific, eid_selection_mask=eids)
        covariate_datasets = [ds, mask_ds]
        label_datasets = [TabularDataset(filepath, self.duration, eid_selection_mask=eids),
                          TabularDataset(filepath, self.event, eid_selection_mask=eids)]

        # make sure we have observations for each label:
        print(split)
        print(label_datasets[1].eid_map[[c for c in label_datasets[1].eid_map.columns if 'event' in c]].sum())

        # oversample if needed:
        if split == 'train' and self.oversampling:
            assert len(self.event) == 1, 'Oversampling only possible for single events.'
            pos_eids = label_datasets[1].eid_map.query(f'{self.event[0]}==1').index.values
            # augment sets:
            for ds_list in [covariate_datasets, label_datasets]:
                for ds in ds_list:
                    pos_ds = pd.concat(10*[ds.eid_map.loc[pos_eids].copy()], axis=0)
                    print(pos_ds.head())
                    pos_ds = pos_ds.reset_index(drop=True)
                    pos_ds.index.name = 'eid'
                    print(pos_ds.head())
                    ds.eid_map = pd.concat([ds.eid_map, pos_ds], axis=0)

            # make sure we have observations for each label:
            print(split)
            print(label_datasets[1].eid_map[[c for c in label_datasets[1].eid_map.columns if 'event' in c]].sum())

        return DatasetWrapper(covariate_datasets, label_datasets)
