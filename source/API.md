# API



```bash
cam
mkdocs new [name]

# update source & page
cd ~/Desktop/nlp-binfo-lab/多标签分类系统/blm-mll
cp /Users/maiqi/Documents/typora/攻读/blm_mll_docs/api.md ./docs/

# open page
cam
mkdocs serve
```

[blm-mll doc](127.0.0.1:8000)



参考：

[weka br](http://waikato.github.io/meka/meka.classifiers.multilabel.BR/)

[sk-multilearn](http://scikit.ml/api/0.1.0/modelselection.html#Performance)

[mkdocs](https://www.mkdocs.org/getting-started/)



## Sequence level

## Problem Transformation

- Binary

- Label Combination

- Pairwise & Threshold methods

- Ensembles of mll



### Ensembles of mll



#### RAkELd

##### Synopsis

Distinct RAndom k-labELsets multi-label classifier.

Divides the label space in to equal partitions of size k, trains a Label Powerset classifier per partition and predicts by summing the result of all trained classifiers.



##### BibTeX

```tex
@ARTICLE{5567103,
    author={G. Tsoumakas and I. Katakis and I. Vlahavas},
    journal={IEEE Transactions on Knowledge and Data Engineering},
    title={Random k-Labelsets for Multilabel Classification},
    year={2011},
    volume={23},
    number={7},
    pages={1079-1089},
    doi={10.1109/TKDE.2010.164},
    ISSN={1041-4347},
    month={July},
}
```



##### Options

- `-ml <method>`

  All blm predictors but `CRF` can serve as base single-label classifier of the selected multi-label classifier. Valid methods are listed here: `SVM`, `RF`, `CNN`, `LSTM`, `GRU`, `Transformer`, `Weighted-Transformer`, `Reformer`

- `-mll_ls <size>` `--RAkEL_labelset_size <size>` should be within [1, labelset_size)

  the desired size of each of the partitions, parameter k according to paper





#### RAkELo

##### Synopsis

Overlapping RAndom k-labELsets multi-label classifier

Divides the label space in to m subsets of size k, trains a Label Powerset classifier for each subset and assign a label to an instance if more than half of all classifiers (majority) from clusters that contain the label assigned the label to the instance.



##### BibTeX

```tex
@ARTICLE{5567103,
    author={G. Tsoumakas and I. Katakis and I. Vlahavas},
    journal={IEEE Transactions on Knowledge and Data Engineering},
    title={Random k-Labelsets for Multilabel Classification},
    year={2011},
    volume={23},
    number={7},
    pages={1079-1089},
    doi={10.1109/TKDE.2010.164},
    ISSN={1041-4347},
    month={July},
}
```



##### Options

- `-ml <method>`

  All blm predictors but `CRF` can serve as base single-label classifier of the selected multi-label classifier. Valid methods are listed here: `SVM`, `RF`, `CNN`, `LSTM`, `GRU`, `Transformer`, `Weighted-Transformer`, `Reformer`

- `-mll_ls <size>` ``--RAkEL_labelset_size <size>` should be within [1, labelset_size)

  the desired size of each of the partitions, parameter k according to paper

- `-mll_mc <count>` `--RAkELo_model_count <count>` 

   the desired number of classifiers, parameter m according to paper.
   
   assure that the product of labelset_size and model_count should be larger than the dimension of label space



## Algorithm Adaptation

### Traditional

#### ML-KNN

##### Synopsis



##### BibTeX

```tex
@article{zhang2007ml,
  title={ML-KNN: A lazy learning approach to multi-label learning},
  author={Zhang, Min-Ling and Zhou, Zhi-Hua},
  journal={Pattern recognition},
  volume={40},
  number={7},
  pages={2038--2048},
  year={2007},
  publisher={Elsevier}
}
```



##### Options

- `-mll_k <number>` `--mll_kNN_k <number>` should be within [1, max_neighbors]

  number of neighbours of each input instance to take into account

- `-mll_s <value>` ` --MLkNN_s <value>` default 1.0

  the smoothing parameter controlling the strength of uniform prior

  （set to be 1 which yields the Laplace smoothing）

- `-mll_ifn <number>` `--MLkNN_ignore_first_neighbours <number>`  should be within [1, max_neighbors]

  number of neighbours of each input instance to take into account



#### BRkNNa/BRkNNb

##### Synopsis

Binary Relevance multi-label classifier based on k-Nearest Neighbors method.

The a version of the classifier assigns the labels that are assigned to at least half of the neighbors.



The b version of the classifier assigns the most popular m labels of the neighbors, where m is the average number of labels assigned to the object’s neighbors.



##### BibTeX

```tex
@inproceedings{EleftheriosSpyromitros2008,
   author = {Eleftherios Spyromitros, Grigorios Tsoumakas, Ioannis Vlahavas},
   booktitle = {Proc. 5th Hellenic Conference on Artificial Intelligence (SETN 2008)},
   title = {An Empirical Study of Lazy Multilabel Classification Algorithms},
   year = {2008},
   location = {Syros, Greece}
}
```



##### Options

- `-mll_k <number>` `--mll_kNN_k <number>` should be within [1, max_neighbors]

  number of neighbours of each input instance to take into account



#### MLARAM

##### Synopsis

HARAM: A Hierarchical ARAM Neural Network for Large-Scale Text Classification

This method aims at increasing the classification speed by adding an extra ART layer for clustering learned prototypes into large clusters. In this case the activation of all prototypes can be replaced by the activation of a small fraction of them, leading to a significant reduction of the classification time.



##### BibTeX



##### Options

- `-mll_v <value> ` `--MLARAM_vigilance <value>` should be within [0, 1]

  parameter for adaptive resonance theory networks,  controls how large a hyperbox can be, 1 it is small (no compression), 0 should assume all range. Normally set between 0.8 and 0.999, it is dataset dependent. It is responsible for the creation of the prototypes, therefore training of the network 

- `-mll_t <value>` ` --MLARAM_threshold <value>`  should be within [0, 1)

  controls how many prototypes participate by the prediction, can be changed for the testing phase





## Residue level

- transformed to sequence level problem 
  (by sliding window)