

# Command line tools

处理blm历史文档的几个思想：

- 在文档介绍blm模块时要简略，同时导引用户读论文和blm文档。

  关键是尽可能地在发生这种简略的地方给出链接和导读。

- 在文档的命令行介绍中，要把blm命令行和新系统命令行同时展现，使用户在复杂用例下只需一个新文档就可以查到所需的全部新老命令。即在命令行层面进行新老文档的深度融合。

- 为提高效率，blm命令行会保留原blm manual中的介绍风格。



先概括介绍cmds的介绍结构，再查表

结构：

multi-label cmds

==> 先介绍 cmds in shared blocks，

==> 介绍新算法 cmds



single-label cmds

==> 这部分 和blm完全一致



太多了，会分成两页





## Multi-label Learning Commands

（介绍 BioSeq-BLM_Seq_mll.py & BioSeq-BLM_Res_mll.py

（介绍如何使用，包括 mll 算法命令 ==(在下一节介绍)==  和 single-label 共享的命令 两部分（根据single-label cmds，分模块介绍如何使用？参考tutorial的异同表，顺便在后面引出single-label cmds）

~~（区分层次，Residue level algorithms ==> transformed to sequence level problem 
(by sliding window)，share the same commands between~~



## Multi-label Learning Algorithms

（介绍 mll 算法命令

link to multi-label learning algorithms in blm-mll



### Binary

#### Binary Relevance

##### Synopsis

Transforms a multi-label classification problem with L labels into L single-label separate binary classification problems using the same base classifier provided in the constructor. The prediction output is the union of all per label classifiers



##### BibTeX



##### Options

- `-mll BR`

- `-ml <method>`

  All blm predictors but `CRF` can serve as base single-label classifier of the selected multi-label classifier. Valid methods are listed here: `SVM`, `RF`, `CNN`, `LSTM`, `GRU`, `Transformer`, `Weighted-Transformer`, `Reformer`



#### Classifier Chain

##### Synopsis



##### BibTeX



##### Options

- `-mll CC`

- `-ml <method>`



### Label Combination

#### LabelPowerSet

##### Synopsis

##### BibTeX



##### Options

- `-mll LP`

- `-ml <method>`

  All blm predictors but `CRF` can serve as base single-label classifier of the selected multi-label classifier. Valid methods are listed here: `SVM`, `RF`, `CNN`, `LSTM`, `GRU`, `Transformer`, `Weighted-Transformer`, `Reformer`



### Pairwise & Threshold methods

#### Calibrated Label Ranking

##### Synopsis



This is an algorithom implemented by MEKA. For more information, please refer to [meka documentation](http://waikato.github.io/meka/meka.classifiers.multilabel.FW/#synopsis).



##### BibTeX



##### Options

- `-mll CLR`

- `-ml <method>`

  All blm predictors but `CRF` can serve as base single-label classifier of the selected multi-label classifier. Valid methods are listed here: `SVM`, `RF`, `CNN`, `LSTM`, `GRU`, `Transformer`, `Weighted-Transformer`, `Reformer`





#### Fourclass Pairwise

##### Synopsis

The Fourclass Pairwise (FW) method. Trains a multi-class base classifier for each pair of labels ($(L*(L-1))/2$ in total), each with four possible class values: {00,01,10,11} representing the possible combinations of relevant (1) /irrelevant (0) for the pair. Uses a voting + threshold scheme at testing time where e.g., 01 from pair jk gives one vote to label k; any label with votes above the threshold is considered relevant. 

This is an algorithom implemented by MEKA. For more information, please refer to [meka documentation](http://waikato.github.io/meka/meka.classifiers.multilabel.FW/#synopsis).



##### BibTeX



##### Options

- `-mll FW`

- `-ml `





#### Rank + Threshold

##### Synopsis

Duplicates multi-label examples into examples with one label each (one vs. rest). Trains a multi-class classifier, and uses a threshold to reconstitute a multi-label classification.

This is an algorithom implemented by MEKA. For more information, please refer to [meka documentation](http://waikato.github.io/meka/meka.classifiers.multilabel.FW/#synopsis).

##### BibTeX



##### Options

- `-mll RT`

- `-ml `



### Ensembles of mll

ensemble of LP, partition strategy …



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

- `-mll RAkELd`

  set the multi-label learning algorithm as RAkELd.

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

- `-mll RAkELo`

  set the multi-label learning algorithm as RAkELo.

- `-ml <method>`

  All blm predictors but `CRF` can serve as base single-label classifier of the selected multi-label classifier. Valid methods are listed here: `SVM`, `RF`, `CNN`, `LSTM`, `GRU`, `Transformer`, `Weighted-Transformer`, `Reformer`

- `-mll_ls <size>` ``--RAkEL_labelset_size <size>` should be within [1, labelset_size)

  the desired size of each of the partitions, parameter k according to paper

- `-mll_mc <count>` `--RAkELo_model_count <count>` 

  the desired number of classifiers, parameter m according to paper.

  assure that the product of labelset_size and model_count should be larger than the dimension of label space



### Adaptation of kNN

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

- `-mll MLkNN`

  set the multi-label learning algorithm as ML-kNN.

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

- `-mll <algorithm>`

  set the multi-label learning algorithm as BRkNNa or BRkNNb.

- `-mll_k <number>` `--mll_kNN_k <number>` should be within [1, max_neighbors]

  number of neighbours of each input instance to take into account



### Adaptation of Neural Network

#### MLARAM

##### Synopsis

HARAM: A Hierarchical ARAM Neural Network for Large-Scale Text Classification

This method aims at increasing the classification speed by adding an extra ART layer for clustering learned prototypes into large clusters. In this case the activation of all prototypes can be replaced by the activation of a small fraction of them, leading to a significant reduction of the classification time.



##### BibTeX



##### Options

- `-mll MLARAM`

  set the multi-label learning algorithm as MLARAM.

- `-mll_v <value> ` `--MLARAM_vigilance <value>` should be within [0, 1]

  parameter for adaptive resonance theory networks,  controls how large a hyperbox can be, 1 it is small (no compression), 0 should assume all range. Normally set between 0.8 and 0.999, it is dataset dependent. It is responsible for the creation of the prototypes, therefore training of the network 

- `-mll_t <value>` ` --MLARAM_threshold <value>`  should be within [0, 1)

  controls how many prototypes participate by the prediction, can be changed for the testing phase.





## 和 single-label 共享的命令 两部分

要利用到下面的单标记



## Single-label Learning Commands

### Intro

来源：说明是blm命令的迁移版本

内容：格式不同，

作用：单独单标记，同时可以用mll的共享文档（怎么说？）



### FeatureExtractionSeq.py

#### Synopsis

“FeatureExtractionSeq.py” is executive python script used for generating feature vectors based on biological language models at sequence level. For more details, please refer to the manual of BLM [3.2 Scripts for feature extraction based on BLMs](https://blm-mll.readthedocs.io/en/latest/Command%20line%20tools.html#id4)



#### Required Options

- `-category {DNA,RNA,Protein}`

  The category of input sequences. 

- `-mode{OHE,BOW,TF-IDF,TR,WE,TM,SR,AF} `

  The feature extraction mode for input sequence which analogies with NLP, for example: bag of words (BOW).

- `-seq_file[SEQ_FILE [SEQ_FILE ...]] `

  The input file in FASTA format. 

- `-label [LABEL [LABEL ...]] `

  The corresponding label of input sequence files.



#### Optional Options

- `-h, --help `

  Show this help message and exit.

- `-words {Kmer,RevKmer,Mismatch,Subsequence,Top-NGram,DR,DT} `

  If you select mode in ['BOW', 'TF-IDF', 'TR', 'WE', 'TM'], you should select word for corresponding mode, for example Mismatch. Pay attention to that different category has different words, please reference to manual.

- `-method METHOD`

  If you select mode in ['OHE', 'WE', 'TM', 'SR', 'AF'], you should select method for corresponding mode, for example, select 'LDA' for 'TM' mode, select 'word2vec' for 'WE' mode and so on. For different category, the methods belong to 'OHE' and 'SR' modeis different, please reference to manual. 

- `-auto_opt {0,1,2}`

  Choose whether automatically traverse the argument list. 2 is automatically traversing the argument list set ahead, 1 is automatically traversing the argument list in a smaller range, while 0 is not (default=0). 

- `-cpu CPU`

  The maximum number of CPU cores used for multiprocessing in generating frequency profile and the number of CPU cores used for multiprocessing during parameter selection process (default=1).

- `-pp_file PP_FILE `

  The physicochemical properties file user input. If input nothing, the default physicochemical properties is: DNA dinucleotide: Rise, Roll, Shift, Slide, Tilt, Twist. DNA trinucleotide: Dnase I, Bendability (DNAse). RNA: Rise, Roll, Shift, Slide, Tilt, Twist. Protein: Hydrophobicity, Hydrophilicity, Mass.

- `-word_size[WORD_SIZE [WORD_SIZE ...]]`

  The word size of sequences for specific words (the range of word_size is between 1 and 6).

- `-mis_num[MIS_NUM [MIS_NUM ...]] `

  For Mismatch words. The max value inexact matching, mis_num should smaller than word_size (the range of mis_num is between 1 and 6).

- `-delta [DELTA [DELTA ...]]`

  For Subsequence words. The value of penalized factor (the range of delta is between 0 and 1).

- `-top_n [TOP_N [TOP_N ...]]`

  The maximum distance between structure statuses (the range of delta is between 1 and 4). It works with Top-n-gram words.

- `-max_dis[MAX_DIS [MAX_DIS ...]] `

  The max distance value for DR words and DT words (default is from 1 to 4).

- `-alpha ALPHA `

  Damping parameter for PageRank used in 'TR' mode, default=0.85.

- `-win_size WIN_SIZE`

  The maximum distance between the current and predicted word within a sentence for ‘word2vec’ in ‘WE’ mode, etc.

- `-vec_dim VEC_DIM`

  The output dimension of feature vectors for 'Glove' model and dimensionality of a word vectors for 'word2vec' and 'fastText' method.

- `-sg SG`

  Training algorithm for 'word2vec' and 'fastText' method. 1 for skip-gram, otherwise CBOW.

- `-in_tm{BOW,TF-IDF,TextRank} `

  While topic model implement subject extraction from a text, the text need to be preprocessed by one of mode in choices.

- `-com_prop COM_PROP`

  If choose topic model mode, please set component proportion for output feature vectors.

- `-oli {0,1}`

  Choose one kind of Oligonucleotide(default=0): 0 represents dinucleotid; 1 represents trinucleotide. For MAC, GAC, NMBAC methods of 'SR' mode.

- `-lag [LAG [LAG ...]]`

  The value of lag (default=1). For DACC, TACC, ACC, ACC-PSSM, AC-PSSM or CC-PSSM methods and so on.

- `-lamada[LAMADA [LAMADA ...]] `

  The value of lamada (default=1). For MAC, PDT, PDT-Profile, GAC or NMBAC methods and so on

- `-w [W [W ...]] `

  The value of weight (default=0.1). For ZCPseKNC method.

- `-k [K [K ...]]`

  The value of Kmer, it works only with ZCPseKNC method.

- `-n [N [N ...]]`

  The maximum distance between structure statuses (default=1). It works with PDT-Profile method.

- `-ui_file UI_FILE `

  The user-defined physicochemical property file.

- `-all_index `

  Choose all physicochemical indices.

- `-no_all_index `

  Do not choose all physicochemical indices, default.

- `-in_af `

  Choose the input for 'AF' mode from 'OHE' mode.

- `-lr LR `

  The value of learning rate, it works only with 'AF' mode.

- `-epochs EPOCHS `

  The epoch number of train process for 'AF' mode.

- `-batch_size BATCH_SIZE `

  The size of mini-batch, it works only with 'AF' mode.

- `-dropout DROPOUT `

  The value of dropout prob, it works only with 'AF' mode.

- `-fea_dim FEA_DIM `

  The output dimension of feature vectors, it works only with 'AF' mode.

- `-hidden_dim HIDDEN_DIM `

  The size of the intermediate (a.k.a., feed forward) layer, it works only with 'AF' mode.

- `-n_layer N_LAYER `

  The number of units for LSTM and GRU, it works only with 'AF' mode.

- `-motif_database {ELM,Mega} `

  The database where input motif file comes from.

- `-motif_file MOTIF_FILE `

  The short linear motifs from ELM database or structural motifs from the MegaMotifBase.

- `-score {ED,MD,CD,HD,JSC,CS,PCC,KLD,none}`

  Choose whether calculate semantic similarity score and what method for calculation.

- `-cv {5,10,j} `

  The cross validation mode. 5 or 10: 5-fold or 10-fold cross validation, j: (character 'j') jackknife cross validation.

- `-fixed_len FIXED_LEN`

  The length of sequence will be fixed via cutting orpadding. If you don't set value for 'fixed_len', it will be the maximum length of all input sequences.

- `-format {tab,svm,csv,tsv} `

  The output format (default=csv). tab –Simple format, delimited by TAB. svm -- The libSVM training data format. csv, tsv -- The format that can be loaded into a spreadsheet program.

- `-bp {0,1}`

  Select use batch mode or not, the parameter will change the directory for generating file based on the method you choose.



### FeatureExtractionRes.py

#### Synopsis

“FeatureExtractionRes.py” is executive python script used for generating feature vectors based on biological language models at residue level.  For more details, please refer to the manual of BLM [3.2 Scripts for feature extraction based on BLMs](https://blm-mll.readthedocs.io/en/latest/Command%20line%20tools.html#id4)



#### Required Options

- `-category {DNA,RNA,Protein}` 

  The category of input sequences. 

- `-method `

  Please select feature extraction method for residue level analysis. 

- `-seq_file` 

  The input sequence file in FASTA format. 

- `-label_file` 

  The corresponding label file. 

  

#### Optional Options

- `-h, --help `

  Show this help message and exit. 

- `-trans {0,1} `

  Select whether use sliding window technique to transform sequence-labelling question to classification question. 

- `-window WINDOW `

  The window size when construct sliding window technique for allocating every label a short sequence. 

- `-fragment {0,1} `

  Please choose whether use the fragment method, 1 is yes while 0 is no.

- `-cpu CPU `

  The maximum number of CPU cores used for multiprocessing in generating frequency profile.

- `-pp_file PP_FILE `

  The physicochemical properties file user input. If input nothing, the default physicochemical properties is: DNA dinucleotide: Rise, Roll, Shift, Slide, Tilt, Twist. DNA trinucleotide: Dnase I, Bendability (DNAse). RNA: Rise, Roll, Shift, Slide, Tilt, Twist. Protein: Hydrophobicity, Hydrophilicity, Mass.

- `-fixed_len FIXED_LEN`

  The length of sequence will be fixed via cutting or padding. If you don't set value for 'fixed_len', it will be the maximum length of all input sequences.

- `-format {tab,svm,csv,tsv} `

  The output format (default=csv). tab –Simple format, delimited by TAB. svm -- The libSVM training data format. csv, tsv -- The format that can be loaded into a spreadsheet program.

- `-bp {0,1} `

  Select use batch mode or not, the parameter will change the directory for generating file based on the method you choose.



### FeatureAnalysis.py

#### Synopsis

“FeatureAnalysis.py” is an executive Python script used for feature analysis. For more details, please refer to the manual of BLM [3.3.1 Method for results analysis]()



#### Required Options

- `-vec_file[VEC_FILE [VEC_FILE ...]] `

  The input feature vector files.

- `-label [LABEL [LABEL ...]] `

  The corresponding label of input vector files is required.

#### Optional Options

- `-h, --help `

  Show this help message and exit.

- `-sn {min-max-scale,standard-scale,L1-normalize,L2-normalize,none}`

  Choose method of standardization or normalization for feature vectors.

- `-cl {AP,DBSCAN,GMM,AGNES,Kmeans,none} `

  Choose method for clustering.

- `-cm {feature,sample} `

  The mode for clustering.

- `-nc NC `

  The number of clusters.

- `-fs{chi2,F-value,MIC,RFE,Tree,none} `

  Select feature select method.

- `-nf NF `

  The number of features after feature selection.

- `-dr{PCA, KernelPCA,TSVD,none} `

  Choose method for dimension reduction.

- `-np NP `

  The dimension of main component after dimension reduction.

- `-rdb {no,fs,dr}`

  Reduce dimension by: 'no'---none; 'fs'---apply feature selection to parameter selection procedure; 'dr'--- apply dimension reduction to parameter selection procedure.

- `-format {tab,svm,csv,tsv}`

  The output format (default=csv). tab – Simple format, delimited by TAB. svm -- The libSVM training data format. csv, tsv -- The format that can be loaded into a spreadsheet program.



### **MachineLearningSeq.py**

#### Synopsis

“MachineLearningSeq.py” is an executive python script used for training predictors and evaluating their performance based on the input benchmark datasets. For more details, please refer to the manual of BLM [3.4 Scripts for machine learning algorithms]()



#### Required Options

- `-ml {SVM,RF,CNN,LSTM,GRU,Transformer,WeightedTransformer,Reformer}`

  The machine-learning algorithm for constructing predictor, for example: Support Vector Machine (SVM).

- `-vec_file[VEC_FILE [VEC_FILE ...]] `

  The input feature vector files.

- `-label [LABEL [LABEL ...]] `

  The corresponding label of input vector files is required.



#### Optional Options

- `-h, --help `

  Show this help message and exit.

- `-cpu CPU`

  The number of CPU cores used for multiprocessing during parameter selection process (default=1).

- `-grid [{0,1} [{0,1} ...]] `

  Grid=0 for rough grid search, grid=1 for meticulous grid search.

- `-cost [COST [COST ...]] `

  Regularization parameter of 'SVM'.

- `-gamma[GAMMA [GAMMA ...]] `

  Kernel coefficient for 'rbf' of 'SVM'.

- `-tree [TREE [TREE ...]] `

  The number of trees in the forest for 'RF'.

- `-lr LR `

  The value of learning rate for deep learning.

- `-epochs EPOCHS `

  The epoch number for train deep model.

- `-batch_size BATCH_SIZE `

  The size of mini-batch for deep learning.

- `-dropout DROPOUT `

  The value of dropout prob for deep learning.

- `-hidden_dim HIDDEN_DIM `

  The size of the intermediate (a.k.a., feed forward) layer.

- `-n_layer N_LAYER `

  The number of units for 'LSTM' and 'GRU'.

- `-out_channels OUT_CHANNELS `

  The number of output channels for 'CNN'

- `-kernel_size KERNEL_SIZE `

  The size of stride for 'CNN'.

- `-d_model D_MODEL`

  The dimension of multi-head attention layer for Transformer or Weighted-Transformer.

- `-d_ff D_FF `

  The dimension of fully connected layer of Transformer or Weighted-Transformer.

- `-heads HEADS `

  The number of heads for Transformer or Weighted-Transformer.

- `-metric {Acc,MCC,AUC,BAcc,F1} `

  The metric for parameter selection

- `-cv {5,10,j}`

  The cross validation mode. 5 or 10: 5-fold or 10-fold cross validation, j: (character 'j') jackknife cross validation.

- `-sp {none,over,under,combine} `

  Select technique for oversampling.

- `-ind_vec_file [IND_VEC_FILE [IND_VEC_FILE ...]]`

  The feature vector files of independent test dataset.

- `-format {tab,svm,csv,tsv}`

  The input format (default=csv). tab –Simple format, delimited by TAB. svm --The libSVM training data format. csv, tsv --The format that can be loaded into a spreadsheet program.



### **MachineLearningRes.py**

#### Synopsis

“MachineLearningRes.py” is an executive python script used for training predictors and evaluating their performance based on the input benchmark datasets. For more details, please refer to the manual of BLM [3.4 Scripts for machine learning algorithms]()



#### Required Options

- `-ml {SVM,RF,CRF,CNN,LSTM,GRU,Transformer,Weighted-Transformer,Reformer}`

  The machine-learning algorithm for constructing predictor, for example: Support Vector Machine (SVM).

- `-vec_file[VEC_FILE [VEC_FILE ...]]`

  The input feature vector file(s). If dichotomy, inputting positive sample before negative sample is required.

- `-label_file `

  The corresponding label file is required.



#### Required Options

- `-h, --help `

  Show this help message and exit.

- `-cpu CPU`

  The number of CPU cores used for multiprocessing during parameter selection process (default=1).

- `-grid [{0,1} [{0,1} ...]] `

  Grid=0 for rough grid search, grid=1 for meticulous grid search.

- `-cost [COST [COST ...]] `

  Regularization parameter of 'SVM'.

- `-gamma[GAMMA [GAMMA ...]] `

  Kernel coefficient for 'rbf' of 'SVM'.

- `-tree [TREE [TREE ...]] `

  The number of trees in the forest for 'RF'.

- `-lr LR `

  The value of learning rate for deep learning.

- `-epochs EPOCHS `

  The epoch number for training deep learning model.

- `-batch_size BATCH_SIZE `

  The size of mini-batch for deep learning.

- `-dropout DROPOUT `

  The value of dropout prob for deep learning.

- `-hidden_dim HIDDEN_DIM `

  The size of the intermediate (a.k.a., feed forward) layer.

- `-n_layer N_LAYER `

  The number of units for 'LSTM' and 'GRU'.

- `-out_channels OUT_CHANNELS `

  The number of output channels for 'CNN'

- `-kernel_size KERNEL_SIZE `

  The size of stride for 'CNN'.

- `-d_model D_MODEL`

  The dimension of multi-head attention layer for Transformer or Weighted-Transformer.

- `-d_ff D_FF `

  The dimension of fully connected layer of Transformer or Weighted-Transformer.

- `-heads HEADS `

  The number of heads for Transformer or Weighted-Transformer.

- `-metric {Acc,MCC,AUC,BAcc,F1} `

  The metric for parameter selection

- `-cv {5,10,j}`

  The cross validation mode. 5 or 10: 5-fold or 10-fold cross validation, j: (character 'j') jackknife cross validation.

- `-sp {none,over,under,combine} `

  Select technique for oversampling.

- `-ind_vec_file [IND_VEC_FILE [IND_VEC_FILE ...]] `

  The feature vector files of independent test dataset.

- `-ind_label_file IND_LABEL_FILE `

  The corresponding label file of independent test dataset.

- `-format {tab,svm,csv,tsv}`

  The input format (default=csv). tab –Simple format, delimited by TAB. svm --The libSVM training data format. csv, tsv --The format that can be loaded into a spreadsheet program.





### BioSeq-BLM_Seq.py

#### Synopsis

“BioSeq-BLM_Seq.py” is executive Python script used for achieving the one-stop
function at sequence level. For more details, please refer to the manual of BLM [3.1 Directory structure]()



#### Required Options

- `-category {DNA,RNA,Protein} `

  The category of input sequences.

- `-mode{OHE,BOW,TF-IDF,TR,WE,TM,SR,AF}`

  The feature extraction mode for input sequence which analogies with NLP, for example: bag of words (BOW).

- `-ml {SVM,RF,CNN,LSTM,GRU,Transformer,WeightedTransformer,Reformer}`

  The machine-learning algorithm for constructing predictor, for example: Support Vector Machine (SVM).

- `-seq_file[SEQ_FILE [SEQ_FILE ...]] `

  The input file in FASTA format.

- `-label [LABEL [LABEL ...]] `

  The corresponding label of input sequence files.



#### Optional Options

- `-h, --help`

  Show this help message and exit.

- `-score {ED,MD,CD,HD,JSC,CS,PCC,KLD,none}`

  Choose whether calculate semantic similarity score and what method for calculation.

- `-words {Kmer,RevKmer,Mismatch,Subsequence,Top-NGram,DR,DT}`

  If you select mode in ['BOW', 'TF-IDF', 'TR', 'WE', 'TM'], you should select word for corresponding mode, for example Mismatch. Pay attention to that different category has different words, please reference to manual.

- `-method METHODIf `

  you select mode in ['OHE', 'WE', 'TM', 'SR', 'AF'], you should select method for corresponding mode, for example, select 'LDA' for 'TM' mode, select 'word2vec' for 'WE' mode and so on. For different category, the methods belong to 'OHE' and 'SR' mode is different, please reference to manual.

- `-auto_opt {0,1,2}`

  Choose whether automatically traverse the argument list. 2 is automatically traversing the argument list set ahead, 1 is automatically traversing the argument list in a smaller range, while 0 is not (default=0).

- `-cpu CPU`

  The maximum number of CPU cores used for multiprocessing in generating frequency profile and the number of CPU cores used for multiprocessing during parameter selection process (default=1).

- `-pp_file PP_FILE`

  The physicochemical properties file user input. If input nothing, the default physicochemical properties is: DNA dinucleotide: Rise, Roll, Shift, Slide, Tilt, Twist. DNA trinucleotide: Dnase I, Bendability (DNAse). RNA: Rise, Roll, Shift, Slide, Tilt, Twist. Protein: Hydrophobicity, Hydrophilicity, Mass.

- `-word_size[WORD_SIZE [WORD_SIZE ...]]`

  The word size of sequences for specific words (the range of word_size is between 1 and 6).

- `-mis_num[MIS_NUM [MIS_NUM ...]]`

  For Mismatch words. The max value inexact matching, mis_num should smaller than word_size (the range of mis_num is between 1 and 6).

- `-delta [DELTA [DELTA ...]]`

  For Subsequence words. The value of penalized factor (the range of delta is between 0 and 1).

- `-top_n [TOP_N [TOP_N ...]]`

  The maximum distance between structure statuses (the range of delta is between 1 and 4). It works with Top-n-gram words.

- `-max_dis[MAX_DIS [MAX_DIS ...]] `

  The max distance value for DR words and DT words (default is from 1 to 4).

- `-alpha ALPHA `

  Damping parameter for PageRank used in 'TR' mode, default=0.85.

- `-win_size WIN_SIZE`

  The maximum distance between the current and predicted word within a sentence for ‘word2vec’ in ‘WE’ mode, etc.

- `-vec_dim VEC_DIM`

  The output dimension of feature vectors for 'Glove' model and dimensionality of a word vectors for 'word2vec' and 'fastText' method.

- `-sg SG`

  Training algorithm for 'word2vec' and 'fastText' method. 1 for skip-gram, otherwise CBOW.

- `-in_tm{BOW,TF-IDF,TextRank}`

  While topic model implement subject extraction from a text, the text need to be preprocessed by one of mode in choices.

- `-com_prop COM_PROP`

  If choose topic model mode, please set component proportion for output feature vectors.

- `-oli {0,1}`

  Choose one kind of Oligonucleotide (default=0): 0 represents dinucleotid; 1 represents trinucleotide. For MAC, GAC, NMBAC methods of 'SR' mode.

- `-lag [LAG [LAG ...]]`

  The value of lag (default=1). For DACC, TACC, ACC, ACC-PSSM, AC-PSSM or CC-PSSM methods and so on.

- `-lamada[LAMADA [LAMADA ...]]`

  The value of lamada (default=1). For MAC, PDT, PDT-Profile, GAC or NMBAC methods and so on

- `-w [W [W ...]] `

  The value of weight (default=0.1). For ZCPseKNC method.

- `-k [K [K ...]] `

  The value of Kmer, it works only with ZCPseKNC method.

- `-n [N [N ...]]`

  The maximum distance between structure statuses (default=1). It works with PDT-Profile method.

- `-ui_file UI_FILE `

  The user-defined physicochemical property file.

- `-all_index `

  Choose all physicochemical indices.

- `-no_all_index `

  Do not choose all physicochemical indices, default.

- `-in_af `

  Choose the input for 'AF' mode from 'OHE' mode.

- `-fea_dim FEA_DIM `

  The output dimension of feature vectors, it works only with 'AF' mode.

- `-motif_database {ELM,Mega} `

  The database where input motif file comes from.

- `-motif_file MOTIF_FILE`

  The short linear motifs from ELM database or structural motifs from the MegaMotifBase.

- `-sn {min-max-scale,standard-scale,L1-normalize,L2-normalize,none}`

  Choose method of standardization or normalization for feature vectors.

- `-cl {AP,DBSCAN,GMM,AGNES,Kmeans,none} `

  Choose method for clustering.

- `-cm {feature,sample} `

  The mode for clustering.

- `-nc NC `

  The number of clusters.

- `-fs{chi2,F-value,MIC,RFE,Tree,none} `

  Select feature select method.

- `-nf NF `

  The number of features after feature selection.

- `-dr{PCA, KernelPCA,TSVD,none} `

  Choose method for dimension reduction.

- `-np NP `

  The dimension of main component after dimension reduction.

- `-rdb {no,fs,dr} `

  Reduce dimension by: 'no'---none; 'fs'---apply feature selection to parameter selection procedure; 'dr'--- apply dimension reduction to parameter selection procedure.

- `-grid [{0,1} [{0,1} ...]] `

  Grid=0 for rough grid search, grid=1 for meticulous grid search.

- `-cost [COST [COST ...]] `

  Regularization parameter of 'SVM'.

- `-gamma[GAMMA [GAMMA ...]] `

  Kernel coefficient for 'rbf' of 'SVM'.

- `-tree [TREE [TREE ...]] `

  The number of trees in the forest for 'RF'.

- `-lr LR `

  The value of learning rate for 'AF' mode and deep learning.

- `-epochs EPOCHS `

  The epoch number for train deep model.

- `-batch_size BATCH_SIZE `

  The size of mini-batch for 'AF' mode and deep learning.

- `-dropout DROPOUT `

  The value of dropout prob for 'AF' mode and deep learning.

- `-hidden_dim HIDDEN_DIM `

  The size of the intermediate (a.k.a., feed forward) layer.

- `-n_layer N_LAYER `

  The number of units for 'LSTM' and 'GRU'.

- `-out_channels OUT_CHANNELS `

  The number of output channels for 'CNN'

- `-kernel_size KERNEL_SIZE `

  The size of stride for 'CNN'.

- `-d_model D_MODEL`

  The dimension of multi-head attention layer for Transformer or Weighted-Transformer.

- `-d_ff D_FF `

  The dimension of fully connected layer of Transformer or Weighted-Transformer.

- `-heads HEADS `

  The number of heads for Transformer or Weighted-Transformer.

- `-metric {Acc,MCC,AUC,BAcc,F1} `

  The metric for parameter selection

- `-cv {5,10,j}`

  The cross validation mode. 5 or 10: 5-fold or 10-fold cross validation, j: (character 'j') jackknife cross validation.

- `-sp {none,over,under,combine} `

  Select technique for oversampling.

- `-ind_seq_file [IND_SEQ_FILE [IND_SEQ_FILE ...]]`

  The independent test dataset in FASTA format.

- `-fixed_len FIXED_LEN`

  The length of sequence will be fixed via cutting orpadding. If you don't set value for 'fixed_len', it will be the maximum length of all input sequences.

- `-format {tab,svm,csv,tsv}`

  The output format (default=csv). tab –Simple format, delimited by TAB. svm --The libSVM training data format. csv, tsv --The format that can be loaded into a spreadsheet program.

- `-bp {0,1}`

  Select use batch mode or not, the parameter will change the directory for generating file based on the method you choose.



### BioSeq-BLM_Res.py

#### Synopsis

“BioSeq-BLM_Res.py” is executive Python script used for achieving the one-stop
function at residue level. For more details, please refer to the manual of BLM [3.1 Directory structure]()



#### Required Options

- `-category {DNA,RNA,Protein} `

  The category of input sequences.

- `-method `

  Please select feature extraction method for residue level analysis.

- `-ml {SVM,RF,CRF,CNN,LSTM,GRU,Transformer,WeightedTransformer,Reformer}`

  The machine-learning algorithm for constructing predictor, for example: Support Vector Machine (SVM).

- `-seq_file `

  The input file in FASTA format.

- `-label_file `

  The corresponding label file.



#### Optional Options

- `-h, --help `

  Show this help message and exit.

- `-trans {0,1}`

  Select whether use sliding window technique to transform sequence-labelling question to classification question.

- `-window WINDOW`

  The window size when construct sliding window technique for allocating every label a short sequence.

- `-fragment {0,1} `

  Please choose whether use the fragment method, 1 is yes while 0 is no.

- `-cpu CPU`

  The maximum number of CPU cores used for multiprocessing in generating frequency profile or The number of CPU cores used for multiprocessing during parameter selection process (default=1).

- `-pp_file PP_FILE`

  The physicochemical properties file user input. If input nothing, the default physicochemical properties is: DNA dinucleotide: Rise, Roll, Shift, Slide, Tilt, Twist. DNA trinucleotide: Dnase I, Bendability (DNAse). RNA: Rise, Roll, Shift, Slide, Tilt, Twist. Protein: Hydrophobicity, Hydrophilicity, Mass.

- `-sn {min-max-scale,standard-scale,L1-normalize,L2-normalize,none}`

  Choose method of standardization or normalization for feature vectors.

- `-cl {AP,DBSCAN,GMM,AGNES,Kmeans,none} `

  Choose method for clustering.

- `-cm {feature,sample} `

  The mode for clustering.

- `-nc NC `

  The number of clusters.

- `-fs {chi2,F-value,MIC,RFE,Tree,none} `

  Select feature select method.

- `-nf NF `

  The number of features after feature selection.

- `-dr {PCA, KernelPCA,TSVD,none} `

  Choose method for dimension reduction.

- `-np NP `

  The dimension of main component after dimension reduction.

- `-rdb {no,fs,dr} `

  Reduce dimension by: 'no'---none; 'fs'---apply feature selection to parameter selection procedure; 'dr'--- apply dimension reduction to parameter selection procedure.

- `-grid [{0,1} [{0,1} ...]] `

  Grid=0 for rough grid search, grid=1 for meticulous grid search.

- `-cost [COST [COST ...]] `

  Regularization parameter of 'SVM'.

- `-gamma [GAMMA [GAMMA ...]] `

  Kernel coefficient for 'rbf' of 'SVM'.

- `-tree [TREE [TREE ...]] `

  The number of trees in the forest for 'RF'.

- `-lr LR `

  The value of learning rate for deep learning.

- `-epochs EPOCHS `

  The epoch number for train deep model.

- `-batch_size BATCH_SIZE `

  The size of mini-batch for deep learning.

- `-dropout DROPOUT `

  The value of dropout prob for deep learning.

- `-hidden_dim HIDDEN_DIM `

  The size of the intermediate (a.k.a., feed forward) layer.

- `-n_layer N_LAYER `

  The number of units for 'LSTM' and 'GRU'.

- `-out_channels OUT_CHANNELS `

  The number of output channels for 'CNN'

- `-kernel_size KERNEL_SIZE `

  The size of stride for 'CNN'.

- `-d_model D_MODEL`

  The dimension of multi-head attention layer for Transformer or Weighted-Transformer.

- `-d_ff D_FF `

  The dimension of fully connected layer of Transformer or Weighted-Transformer.

- `-heads HEADS `

  The number of heads for Transformer or Weighted-Transformer.

- `-metric {Acc,MCC,AUC,BAcc,F1} `

  The metric for parameter selection

- `-cv {5,10,j}`

  The cross validation mode. 5 or 10: 5-fold or 10-fold cross validation, j: (character 'j') jackknife cross validation.

- `-sp {none,over,under,combine} `

  Select technique for oversampling.

- `-ind_seq_file IND_SEQ_FILE `

  The independent test dataset in FASTA format.

- `-ind_label_file IND_LABEL_FILE `

  The corresponding label file of independent test dataset.

- `-fixed_len FIXED_LEN`

  The length of sequence will be fixed via cutting orpadding. If you don't set value for 'fixed_len', it will be the maximum length of all input sequences.

- `-format {tab,svm,csv,tsv}`

  The output format (default=csv). tab –Simple format, delimited by TAB. svm --The libSVM training data format. csv, tsv --The format that can be loaded into a spreadsheet program.

- `-bp {0,1}`

  Select use batch mode or not, the parameter will change the directory for generating file based on the method you choose.





## References
