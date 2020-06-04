# leveraging synergy in multiple meteorological datasets with deep learning for rainfall-runoff modeling

Accompanying code for our HESSD paper "A note on leveraging synergy in multiple meteorological datasets with deep learning for rainfall-runoff modeling "

```
Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss., https://doi.org/10.5194/hess-2020-221, in review, 2020.  
```

The manuscript can be found here (publicly available): [A note on leveraging synergy in multiple meteorological datasets with deep learning for rainfall-runoff modeling ](https://www.hydrol-earth-syst-sci-discuss.net/hess-2020-221/)


## Content of the repository

- `data/` contains the list of basins (USGS gauge ids) considered in our study
- `codebase/` contains the implementations of all necessary parts for training and evaluating the models
- `configs/` contains the config files that were used for training our models. These files can be used to reproduce our results, if you want to train the models on your own. Note that you have to change the `data_dir` argument in the config files to point to your local CAMELS US data set folder. You will also need to have the extended forcing products (Maurer and NLDAS), see below.
- `environments/` contains Anaconda environment files that can be used to create a Python environment with all required packages. For details, see below.
- `main.py` Main python file used for training and evaluating of our models
- `run_scheduler.py` Helper file to batch train/evaluate multiple models in a single call on one/multiple GPUs (and one or multiple models per GPU)

- `notebooks/` contain three notebooks, guiding through the results of our study. These notebooks should probably be your starting point.
    - `notebooks/performance.ipynb`: In this notebook, our modeling results are evaluated and compared against the benchmark models. All numbers and figures of the first two subsections of the results can be found here.
    - `notebooks/ranking.ipynb`: In this notebook, you can find the derivation of the feature ranking and the model robustness plot of the third subsection of the results.
    - `notebooks/embedding.ipynb`: In this notebook, you can find the analysis of the catchment embedding learned by our model as well as the cluster analysis. Here you find everything of the last subsection of the results.

## Setup to run the code locally

Download this repository either as zip-file or clone it to your local file system by running

```
git clone git@github.com:kratzert/multiple_forcing.git
```

### Setup Python environment
Within this repository we provide environment files that can be used with Anaconda or Miniconda to create an environment with all required packages. We are working with Linux and provide our exact environments in the `environments/linux` folder. For Windows or Mac use the files in the `raw` subdirectory. Note, we can't vouch for these environments to work.
Our environments require you to have a GPU with CUDA support (different files for either CUDA 9.2 or CUDA 10.1). If you have no GPU available, check the PyTorch homepage on how to install PyTorch without GPU support. The rest of the environment is identical.

To create an environment with Anaconda/Miniconda, run e.g. (for the Linux environment with CUDA 10.1)

```
conda env create -f environments/linux/environment_cuda10_1.yml
```
from the base directory.

## Data needed

### Required Downloads

First of all you need the CAMELS data set, to run any of your code. This data set can be downloaded for free here:

- [CAMELS: Catchment Attributes and Meteorology for Large-sample Studies - Dataset Downloads](https://ral.ucar.edu/solutions/products/camels) Make sure to download the `CAMELS time series meteorology, observed flow, meta data (.zip)` file, as well as the `CAMELS Attributes (.zip)`. Extract the data set on your file system and make sure to put the attribute folder (`camels_attributes_v2.0`) inside the CAMELS main directory.

We trained our models with an updated version of the Maurer and NLDAS forcing data, that is still not published officially (CAMELS data set will be updated soon). The updated versions forcing contain daily minimum and maximum temperature. The original data included in the CAMELS data set only includes daily mean temperature. You can find the updated forcings temporarily here:

- [Updated Maurer forcing with daily minimum and maximum temperature](https://www.hydroshare.org/resource/17c896843cf940339c3c3496d0c1c077/)
- [Updated NLDAS forcing with daily minimum and maximum temperature](https://www.hydroshare.org/resource/0a68bfd7ddf642a8be9041d60f40868c/)

Download and extract the updated forcing into the `basin_mean_forcing` folder of the CAMELS data set and do not rename it (name should be `maurer_extended` and `nldas_extended`).

Next you need the simulations of all benchmark models. These can be downloaded from HydroShare under the following link:

- [CAMELS benchmark models](http://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1)

### Optional Downloads

We will provide our pre-trained models, so you can do evaluations without training the models on your own. Download link will follow.

### Train or evaluate a single model

Before starting to do anything, make sure you have activated the conda environment.

```
conda activate pytorch
```

To train a model, run the following line of code from the terminal

```
python main.py train --config_file PATH/TO/CONFIG.yml
```
where `PATH/TO/CONFIG.yml` can be either the relative or absolute path to the config file you want to use for training.

To evaluate a trained model, run 
```
python main.py evaluate --run_dir PATH/TO/RUN_DIR
```
where `PATH/TO/RUN_DIR` can be either the relative or absolute path to the run directory of the train model. By default, the model will be evaluated on the last available model checkpoint. To evaluate a specific checkpoint at `--epoch NUMBER`, where `NUMBER` is an integer specifying the epoch you want to evaluate.
The results will be stored in the `test` directory within the run directory. The file is a pickle dump of a dictionary, containing the simulations + observations for each basin as an xarray.

### Train or evaluate a multiple models

To facilitate training or evaluating of multiple models on one or optionally multiple GPUs you can use the `run_scheduler.py` script. This will most likely only work on Linux machines, due to how environment variables are used.

The arguments are:
- `--mode` either `train` or `evaluate`
- `--config_dir`, if you chose the train mode, you have to pass the path to a folder, containing the config files you want to use for training. For each YAML file in the specified directory (or its subdirectories) one training run is started.
- `--run_dir`, if you chose the evaluate mode, you have to pass the path to a folder, containing multiple run directories. Each of those runs will be evaluated and the results stored in the respective run directory (see above).
- `--gpu_ids` a list of integers, specifying the GPU ids you want to use for training.
- `--runs_per_gpu` an integer, specifying how many models will be trained/evaluated in parallel on a single GPU

Example for training one model for each config file in a certain directory on 3 GPUs (id 0,1,2) and 3 runs per GPU, the command would be
```
python run_scheduler.py --mode train --config_dir configs/ --gpu_ids 0 1 2 --runs_per_gpu 3
```

Note: To train on model in the fastest possible way (caching the training data in memory) does require a substantial amount of RAM (up to around 60GB for the all forcing model). So be cautious when training multiple models at once. You always have the option to set `cache_data` to `False` in the configs, which will slow down training considerably but does reduce the memory requirements to almost nothing.

### Notebooks

We will soon add notebooks that were used to evaluate all results and produce all numbers and figures used in the manuscript.

## Citation

If you use any of this code in your experiments, please make sure to cite the following publication

```
@Article{kratzert2020synergy,
author = {Kratzert, F. and Klotz, D. and Hochreiter, S. and Nearing, G. S.},
title = {A note on leveraging synergy in multiple meteorological datasets with deep learning for rainfall-runoff modeling},
journal = {Hydrology and Earth System Sciences Discussions},
volume = {2020},
year = {2020},
pages = {1--26},
url = {https://www.hydrol-earth-syst-sci-discuss.net/hess-2020-221/},
doi = {10.5194/hess-2020-221}
}
```

## Acknowledgement

Frederik Kratzert is supported by a Google Faculty Research Award. This study was furthermore supported by Google Cloud Platform grant and parts of the experiments were run on the Google Cloud Platform.

## License of our code
[Apache License 2.0](https://github.com/kratzert/multiple_forcing/blob/master/LICENSE)

## License of the updated Maurer and NLDAS forcings and our pre-trained models
The CAMELS data set only allows non-commercial use. Thus, our pre-trained models and the updated Maurer and NLDAS forcings underlie the same [TERMS OF USE](https://www2.ucar.edu/terms-of-use) as the CAMELS data set. 
