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





### Liunx

> 在实验室GPU服务器可使用后，再来总结

```

```



### OS X

```
conda env remove -n macos_venv2

# 新环境，基础依赖
conda env create -f ana_venv.yaml

# 避免 resolve not found
conda search pkg

# yaml
name: macos_venv2
channels:
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/cloud/conda-forge
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/main/
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/pkgs/free/
  - https://mirrors.sjtug.sjtu.edu.cn/anaconda/cloud/conda-forge/
  - apple
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
  - liac-arff
  - tensorflow-deps==2.6.0
  - pytorch


# tensorflow
# conda install -c apple tensorflow-deps==2.6.0 
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal

# pytorch
# https://anaconda.org/pytorch/pytorch
# conda install -c pytorch pytorch

# keras (tensorflow bundled)
# conda install keras

# scikit-multilearn==0.2.0 bleeding edge
cd xxx/pkg
python setup.py install

# conda install skorch -y

from skmultilearn.ext import download_meka

meka_classpath = download_meka()
meka_classpath


# test env
from skmultilearn.ext import Meka

import tensorflow as tf
print(tf.__version__) # 2.9.2

import keras as k
print(k.__version__)
```



### Windows

同样推荐 anaconda / pip 安装

```
 
```

