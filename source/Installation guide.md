# Installation guide



## Download stand-alone package

```
git clone xx
```



## Set up the environment

> 先说要下载哪些包，版本要求

给出版本建议

If you want to install xx from this repository, you need to install the dependencies first.

Install `pytorch >= 1.0` with the command: `conda install pytorch -c pytorch` or refer to `pytorch` website [https://pytorch.org](https://pytorch.org/).

```

```





### Liunx 20.04

```
# ana_venv.yaml
name: blm_mll_linux
channels:
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/cloud/conda-forge
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/free/
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/cloud/conda-forge/
  - pytorch
dependencies:
	- python==3.8.5
  - pandas
  - tqdm
  - numpy
  - matplotlib
  - scipy
  - scikit-learn
  - imbalanced-learn
  - skorch
  - six
  - joblib
  - docx
  - liac-arff
  - tensorflow-gpu>=2.6.0
  - pytorch
  - networkx
  - gensim

# 新环境，基础依赖
conda env create -f ana_venv.yaml

conda activate blm_mll_linux

# scikit-multilearn bleeding-edge version
git clone https://github.com/scikit-multilearn/scikit-multilearn.git
cd scikit-multilearn

# 修改参数
method=
encoding=

python setup.py install
```



### Windows

同样推荐 anaconda / pip 安装

```
 
```





```bash
mkdir software
cd 

wget https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.13.0+-src.tar.gz
```

