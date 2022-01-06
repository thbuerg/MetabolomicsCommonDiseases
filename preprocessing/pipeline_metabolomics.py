import os
import pathlib
import pandas as pd
import numpy as np
import prefect as pf
import miceforest as mf
from prefect.engine.results import LocalResult
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.serializers import JSONSerializer
from prefect.executors import LocalDaskExecutor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from category_encoders.ordinal import OrdinalEncoder
import pickle


output_directory = '/your/output/dir/'
output_name = 'your_dataset_name'

json_serializer = JSONSerializer()


class ApplyImputer(pf.Task):
    """
    Takes a list of tuples, where the first pos is the eids_dict, the second is the kernel, the third is the split.
    Then applies imputer and saves to file.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_target(self, cv_partition, split):
        """
        Update Target string at runtime.
        :return:
        """
        self.target = f"partition_{cv_partition}/{split}_baseline_imputed.csv"

    def run(self, partition_split_dict):
        """
        split tuple is a tuple in the form of
        (  (partition_idx, eids_dict,  (data_merged, data_merged_description)  ), imputer, split)
        :param partition_split_dict:
        :return:
        """
        split = partition_split_dict['split']
        partition = partition_split_dict["cv_partition"]
        eids = partition_split_dict['eids_dict'][split]

        assert split in ['test', 'train', 'valid']
        self._update_target(partition_split_dict['cv_partition'], split)
        data = partition_split_dict['data'].loc[eids]

        # Save partitions
        data_output_path = f"{output_directory}/{output_name}/partition_{partition}/{split}"
        pathlib.Path(data_output_path).mkdir(parents=True, exist_ok=True)
        data.reset_index().to_feather(f"{data_output_path}/data.feather")

        # Impute data
        with open(partition_split_dict['imputer_path'], "rb") as input_file: imputer = pickle.load(input_file)
        data_imputed = imputer.impute_new_data(new_data=data).complete_data()
        partition_split_dict['data'] = data_imputed

        data_output_path = f"{output_directory}/{output_name}/partition_{partition}/{split}"
        pathlib.Path(data_output_path).mkdir(parents=True, exist_ok=True)
        data_imputed.reset_index().to_feather(f"{data_output_path}/data_imputed.feather")

        return partition_split_dict


class ApplyNorm(pf.Task):
    """
    Takes a list of tuples, where the first pos is the eid_dict, the second is the kernel, the third is the split.
    Then applies imputer and saves to file.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _update_target(self, cv_partition, split):
        """
        Update Target string at runtime.
        :return:
        """
        self.target = f"partition_{cv_partition}/{split}_baseline_imputed_normalized.csv"

    def run(self, partition_split_dict):
        """
        DICT
        :param partition_split_dict:
        :return:
        """
        split = partition_split_dict['split']
        partition = partition_split_dict['cv_partition']
        self._update_target(partition, split)

        description = partition_split_dict['description']

        noncategorical_covariates = description.reset_index() \
            .set_index('dtype').loc[['int', "float"]] \
            .query("(isTarget == False) & (based_on != 'diagnoses_emb') & (based_on != 'eid')")['covariate'].values


        noncat_data = partition_split_dict['data'][noncategorical_covariates].copy()

        # log 1p transform!:
        for c in noncat_data.columns:
            if c.startswith('NMR'):
                noncat_data[c] = np.log1p(noncat_data[c].values)

        noncat_data = noncat_data.values

        noncat_data = pd.DataFrame(partition_split_dict['normalizer'].transform(noncat_data),
                                   columns=noncategorical_covariates)

        for v in noncategorical_covariates:
            partition_split_dict['data'][v] = noncat_data[v].values

        # save preprocessed data
        data_output_path = f"{output_directory}/{output_name}/partition_{partition}/{split}"
        pathlib.Path(data_output_path).mkdir(parents=True, exist_ok=True)
        partition_split_dict['data'].reset_index().to_feather(f"{data_output_path}/data_imputed_normalized.feather")

        # return partition_split_dict
        return partition_split_dict['data']

@pf.task(target="data_merged_dict.p",
         checkpoint=True,
         log_stdout=True,
      result=LocalResult(dir=f"{output_directory}/{output_name}")
      )
def read_and_merge_data(covariate_paths, input_data_dir):
    logger = pf.context.get("logger")
    logger.info("Data")
    data_dfs = [pd.read_feather(f"{input_data_dir}/{covariate_paths[covariate][0]}").set_index("eid") for covariate in covariate_paths]
    data_merged = pd.concat(data_dfs, axis=1).copy()
    output_path = f"{output_directory}/{output_name}"

    data_merged.reset_index().to_feather(f"{output_path}/data_merged.feather")

    logger.info("Descriptions")
    description_dfs = [pd.read_feather(f"{input_data_dir}/{covariate_paths[covariate][1]}") for covariate in covariate_paths]
    description_merged = pd.concat([df if i == 0 else df.tail(-1) for i, df in enumerate(description_dfs)], axis=0).reset_index()
    description_merged.reset_index(drop=True).to_feather(f"{output_path}/description_merged.feather")

    return {"data": data_merged.query('NMR_FLAG==True'), "description": description_merged}

@pf.task(name="encode_categoricals",
      target="data_encoded.p",
      checkpoint=True,
      result=LocalResult(dir=f"{output_directory}/{output_name}")
      )
def encode_categoricals(data_dict):
    logger = pf.context.get("logger")
    data = data_dict["data"]
    description = data_dict["description"]

    cat_cols = [c for c in description.set_index("dtype").loc[["category"]].covariate.to_list() if "date" not in c]

    mapping = [{"col": c, "mapping": {e: i for i, e in enumerate([v for v in data[c].unique().tolist() if v==v])}} for c in cat_cols]
    for i, c in enumerate(cat_cols): mapping[i]["mapping"].update({np.nan: -2})

    enc = OrdinalEncoder(cols=cat_cols, mapping=mapping, handle_missing="return_nan")
    data = enc.fit_transform(data)

    description["mapping"] = np.nan
    for i, c in enumerate(cat_cols):
        description.loc[description.covariate == c, 'mapping'] = str(enc.mapping[i]["mapping"])
        if data[c].nunique() > 2:
            ohe_encoded = pd.get_dummies(data[c], prefix=c)
            data[ohe_encoded.columns] = ohe_encoded
            for col in ohe_encoded.columns:
                description = description.append(
                    {"covariate": col, "dtype": "bool", "isTarget": False,
                     "based_on": description.loc[description.covariate == c, "based_on"].iloc[0],
                     "aggr_fn": np.nan, "mapping": str(enc.mapping[i]["mapping"])}, ignore_index=True)
    description["based_on"] = description["based_on"].astype(str)

    description.reset_index(drop=True).to_feather(f"{output_directory}/{output_name}/description.feather")

    logger.info(f"{len(cat_cols)} columns one-hot-encoded")
    return {"data": data, "description": description}

@pf.task(name="apply_exclusion_criteria",
      target="data_merged_excluded_dict.p",
      checkpoint=True,
      result=LocalResult(dir=f"{output_directory}/{output_name}")
      )
def apply_exclusion_criteria(data_dict, exclusion_criteria):
    logger = pf.context.get("logger")
    data = data_dict["data"]
    data_excl = data.copy().query(exclusion_criteria).reset_index(drop=False).set_index("eid")
    output_path = f"{output_directory}/{output_name}"
    data_excl.reset_index().to_feather(f"{output_path}/data_excl.feather")
    logger.info(f"{len(data)-len(data_excl)} eids excluded")
    return {"data": data, "description": data_dict["description"]}

@pf.task(name="get_eids_for_partitions",
      target=f"eids.json",
      checkpoint=True,
      result=LocalResult(dir=f"{output_directory}/{output_name}", serializer=json_serializer)
      )

def get_eids_for_partitions(data_dict, partition_column, valid_size=0.1):
    logger = pf.context.get("logger")

    data_all = data_dict["data"]
    eids_all = data_all.index.values
    groups = data_all.reset_index().set_index(partition_column).index.value_counts().index.to_list()
    splits = {i: data_all.query(f"{partition_column}==@group").index.tolist() for i, group in enumerate(groups)}

    eids_dict = OrderedDict()
    for partition in range(len(groups)):
        eids_dict[partition] = {}
        eids_test = splits[partition]
        eids_notest = sorted(list(set(eids_all) - set(eids_test)))
        eids_train, eids_valid = train_test_split(eids_notest, test_size=valid_size, shuffle=False)

        if bool(set(eids_train) & set(eids_valid) & set(eids_test)) == True:
            logger.warning(f"Overlap of eids in partition {partition}")
        else:
            logger.info(f"No overlap of eids in partition {partition}")

        eids_dict[partition]["train"] = eids_train
        eids_dict[partition]["valid"] = eids_valid
        eids_dict[partition]["test"] = eids_test

    return eids_dict

@pf.task
def get_partitions(data_dict, eids_dict):
    partition_dicts = [{**data_dict, 'cv_partition': partition_idx, 'eids_dict': eids_dict[partition_idx]} for partition_idx in eids_dict.keys()]
    return partition_dicts


@pf.task(name="fit_imputer",
      target="{task_name}/{task_full_name}_kernel.p",
      checkpoint=True,
      result=LocalResult(dir=os.path.join(output_directory, output_name, "pipeline/"))
      )
def fit_imputer(partition_dict):
    """
    Fit an imputer to train set and pickle it
    (partition_idx, eids_dict, (data, data_descr) )
    """
    eids_train = partition_dict['eids_dict']['train']
    data = partition_dict['data'].loc[eids_train]
    partition = partition_dict["cv_partition"]

    missing = data.columns[data.isna().any()].to_list()
    missing = [col for col in missing if not "NMR_measurement_quality_flagged" in col]

    events = [col for col in data.columns if "_event" in col]

    variable_schema = {}
    for m in missing:
        variable_schema[m] = [x for x in missing if x != m]+events
    kernel = mf.KernelDataSet(data,
                              variable_schema=variable_schema,
                              save_all_iterations=True,
                              random_state=42)

    # Run the MICE algorithm for 3 iterations
    kernel.mice(3, n_jobs=1, n_estimators=8,
                max_features="sqrt", bootstrap=True, max_depth=8, verbose=True)

    data_output_path = f"{output_directory}/{output_name}/partition_{partition}"
    pathlib.Path(data_output_path).mkdir(parents=True, exist_ok=True)

    imputer_path = f"{data_output_path}/imputer.p"
    with open(imputer_path, "wb") as output_file: pickle.dump(kernel, output_file)
    del kernel
    return imputer_path

@pf.task
def get_splits_per_partition(partition_dict, imputer_path, splits):
    partition_split_dicts = [{**partition_dict, 'imputer_path': imputer_path, 'split': s} for s in splits]
    return partition_split_dicts

@pf.task(name="fit_normalization",
         target="{task_name}/{task_full_name}_norm.p",
      checkpoint=True,
      result=LocalResult(dir=os.path.join(output_directory, output_name, "pipeline/"))
      )
def fit_normalization(partition_split_dicts):
    """
    Fit an imputer to train set and pickle it.

    imputed_tuples should be a list of dicts of the form:
    data_imputed is the imputed data for a split in the partition for partition idx

    """
    # first get vars:
    description = partition_split_dicts[0]['description']
    noncategorical_covariates = description.reset_index() \
        .set_index('dtype').loc[['int', "float"]] \
        .query("(isTarget == False) & (based_on != 'diagnoses_emb') & (based_on != 'eid')")['covariate'].values

    # fit normalizer for each train split:
    fitted_normalizers = {}
    for d in partition_split_dicts:
        if d['split'] == 'train':
            if 'eid' in d['data'].columns:
                data = d['data'].set_index('eid')
            else:
                data = d['data']
            noncategorical_data = data[noncategorical_covariates]

            # log 1p transform!:
            for c in noncategorical_data.columns:
                if c.startswith('NMR'):
                    noncategorical_data[c] = np.log1p(noncategorical_data[c].values)

            noncategorical_data = noncategorical_data.values

            norm = StandardScaler(with_mean=True, with_std=True, copy=True).fit(noncategorical_data)
            fitted_normalizers[d['cv_partition']] = norm

    partition_split_dicts = [{**d, 'normalizer': fitted_normalizers[d['cv_partition']]} for d in partition_split_dicts]
    return partition_split_dicts


Impute = ApplyImputer(name="apply_imputer",
                      target=f"partition_23/baseline_imputed.csv",
                      checkpoint=True,
                      result=LocalResult(dir=f"{output_directory}/{output_name}/cv_partitions/"),
                                         # serializer=pd_serializer)
                      )

Normalize = ApplyNorm(name="apply_norm",
                      target=f"partition_23/baseline_imputed_normalized.csv",
                      checkpoint=True,
                      result=LocalResult(dir=f"{output_directory}/{output_name}/cv_partitions/"),
                                         # serializer=pd_serializer)
                      )

with pf.Flow("ukb_pipeline") as flow:
    input_data_dir = pf.Parameter('input_data',
                                  default=f'{output_name}/2_datasets_pre/210709_metabolomics/')
    partition_column = pf.Parameter('partition_column', default="uk_biobank_assessment_centre")
    valid_size = pf.Parameter('valid_size', default=0.1)
    data_filenames = {
        "covariates": ("baseline_covariates.feather", "baseline_covariates_description.feather"),
        "pgs": ("baseline_pgs.feather", "baseline_pgs_description.feather"),
        "endpoints": ("baseline_endpoints.feather", "baseline_endpoints_description.feather"),
    }


    data_dict = read_and_merge_data(data_filenames, input_data_dir)
    data_dict = encode_categoricals(data_dict)
    eids_dict = get_eids_for_partitions(data_dict, partition_column=partition_column, valid_size=valid_size)
    partition_dicts = get_partitions(data_dict, eids_dict)

    # fit imputer per partition
    imputer_paths = fit_imputer.map(partition_dict=partition_dicts)

    partition_split_dicts = get_splits_per_partition.map(partition_dicts,
                                                     imputer_paths,
                                                     splits=pf.unmapped(['train', 'test', 'valid'])
                                                     )

    partition_split_dicts = Impute.map(partition_split_dict=pf.flatten(partition_split_dicts))
    partition_split_dicts = fit_normalization(partition_split_dicts=partition_split_dicts)

    normalized = Normalize.map(partition_split_dict=partition_split_dicts)

if __name__ == "__main__":
    flow.executor = LocalDaskExecutor(scheduler="threads", num_workers=60)

    # run locally
    runner = FlowRunner(flow=flow)
    flow_state = runner.run(return_tasks=flow.tasks)
