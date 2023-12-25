# Installation Guide

## Download stand-alone package

clone the repository using Git or checkout with SVN.

```bash
git clone https://github.com/tonqletSpace/BioSeq-BLM-mllearn
```



## Set up the environment

Before using the system, Python engine and some core dependencies need be installed and configured first. We recommend to install Python 3.x 64-bit (especially 3.8 64-bit) engine and use [Anaconda](https://www.anaconda.com/download) to manage the entire Python environment. We supply a configuration file as follow which specifies dependencies our system requires minimally, and name it `venv_initial.yaml`.

```bash
# venv_initial.yaml
name: blm_mll_linux # virtual environment name
channels:
	# add channels or mirrors to improve speed
  # - pytorch
  
dependencies:
  - python==3.8.5
  - numpy
  - matplotlib
  - scipy
  - scikit-learn
  - imbalanced-learn
  - pytorch>=1.0
  - pandas
  - tqdm
  - skorch
  - gensim
  - networkx
```

Use anaconda command lines to create virtual environment with the configuration file.

```bash
# create virtual Python environment from configuration file.
conda env create -f venv_initial.yaml

# conda activate blm_mll_linux
conda activate mll_ux_test
```

Our system depends on [scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn) module and need to install the bleeding-edge version of because the latest release have some essential bugs. To install the bleeding-edge version of scikit-multilearn, clone this repository and run `setup.py`

```bash
git clone https://github.com/scikit-multilearn/scikit-multilearn.git
cd scikit-multilearn

python setup.py install
```
