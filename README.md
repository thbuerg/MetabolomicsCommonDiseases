<div align="center">

![Logo](./src/msm_logo.png?raw=true "Logo")

⛑ **Metabolomic profiles predict individual multi-disease outcomes** ⛑

[comment]: <> (<!--)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thbuerg/MetabolomicsCommonDiseases/blob/main/analysis/examples/MetabolomicsInference.ipynb)
[![Paper](https://img.shields.io/badge/Paper-Nature%20Medicine-red)](https://www.nature.com/articles/s41591-022-01980-3)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6200202.svg)](https://doi.org/10.5281/zenodo.6200202)
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

[comment]: <> (-->)

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
We provide you a ready-to-use [Google colab notebook](https://colab.research.google.com/github/thbuerg/MetabolomicsCommonDiseases/blob/main/analysis/examples/MetabolomicsInference.ipynb) with a trained version of our MetabolomicStateModel. Upload your dataset of Nightingale NMR metabolomics and run the model!  
 
**NOTE**: Data must be provided in [this format](https://github.com/thbuerg/MetabolomicsCommonDiseases/blob/main/analysis/examples/sample.csv).  
 
**DISCLAIMER**: This model is intended for research use only. We provide the NMR normalization pipeline as fitted on UK Biobank. Cohort-specific rescaling might be advisable.

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
cd source

# run training
bash run/run_MSM.sh
```
 
## License
This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## Citation   
```
@article{buergel2022metabolomic,
  title={Metabolomic profiles predict individual multidisease outcomes},
  author={Buergel, Thore and Steinfeldt, Jakob and Ruyoga, Greg and Pietzner, Maik and Bizzarri, Daniele and Vojinovic, Dina and Upmeier zu Belzen, Julius and Loock, Lukas and Kittner, Paul and Christmann, Lara and others},
  journal={Nature Medicine},
  pages={1--12},
  year={2022},
  publisher={Nature Publishing Group}
}
```  
