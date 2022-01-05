<div align="center">    

![Logo](./src/msm_logo.png?raw=true "Logo")


⛑**Metabolomic profiles predict individual multi-disease outcomes in the UK Biobank cohort**⛑

<!--
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->

</div>
 
## Description   
Code related to the paper "Metabolomic profiles predict individual multi-disease outcomes in the UK Biobank cohort". 
This repo is a python package for preprocessing UK Biobank data and preprocessing, training and evaluating the proposed MetabolomicStateModel score.

![Workflow](./src/fig1.png?raw=true "Workflow")

## Methods
The **MetabolomicStateModel** is based on [DeepSurv](https://arxiv.org/abs/1606.00931) (the original implementation can be found [here](https://github.com/jaredleekatzman/DeepSurv)). Using a residual neural network, it learns a shared-representation of the NMR metabolomics data to predict log partial hazards for common disease endpoints.

![Architecture](./src/fig2.png?raw=true "Architecture")

## Assets
This repo contains code to preprocess [UK Biobank](https://www.ukbiobank.ac.uk/) data, train the MetabolomicStateModel and analyze/evaluate its performance.

- Preprocessing involves parsing primary care records for desired diagnosis. 
- Training involves Model specification via pytorch-lightning and hydra.
- Evaluation involves extensive benchmarks with linear Models, and calculation of bootstrapped metrics.
- Visualization contains the code to generate the figures displayed in the paper. 

## Use the MetabolomicStateModel on your data
We provide you a ready-to-use [Google colab notebook](https://www.google.com) with a trained version of our MetabolomicStateModel. Upload you batch of Nightingale NMR metabolomics data and run the model!

## How to train the MetabolomicStateModel  
1. First, install dependencies   
```bash
# clone project   
git clone https://github.com/thbuerg/MetabolomicsCommonDiseases

# install project   
cd MetabolomicsCommonDiseases
pip install -e .   
pip install -r requirements.txt
 ```   

2. Download UK Biobank data. Execute preprocessing notebooks on the downloaded data.

3. Set up [Neptune.ai](https://www.neptune.ai)

4. Edit the `config.yaml` in `metabolomicstatemodel/run/config/`:
```yaml
data_dir: /path/to/data
code_dir: /path/to/repo_base
setup:
  project: <YourNeptuneWorkspace>/<YourProject>
experiment:
  tabular_filepath: /path/to/processed/data
```

5. Train the NeuralCVD Model (make sure you are on a machine w/ GPU)
 ```bash
# module folder
cd metabolomicstatemodel

# run training
bash run/run_MSM.sh
```

## Citation   
```
@article{thisonecoolstory,
  title={Neural network-based integration of polygenic and clinical information: Development and validation of a prediction model for 10 year risk of major adverse cardiac events in the UK Biobank cohort},
  author={Jakob Steinfeldt, Thore Buergel},
  journal={tbd},
  year={2022}
}
```  
