

# Command line tools

处理blm历史文档的几个思想：

- 在文档介绍blm模块时要简略，同时导引用户读论文和blm文档。

  关键是尽可能地在发生这种简略的地方给出链接和导读。

- 在文档的命令行介绍中，要把blm命令行和新系统命令行同时展现，使用户在复杂用例下只需一个新文档就可以查到所需的全部新老命令。即在命令行层面进行新老文档的深度融合。

- 为提高效率，blm命令行会保留原blm manual中的介绍风格



This page introduces command lines for both the multi-label learning flow and the single-label learning flow in out proposed system blm-mll. Moreover, all multi-label learning algorithms utilized in the systems and their corresbonding options are introduced as well. The command lines of both flow are script-oriented where a script provides adequate options for a general function block of system flow (see in [流程图 tutorial](x) ) or a one-stop fcuntion for a specific task. While the multi-label learning algorithms are introduced algorithm-oriented.

There are three sections in this page:

- Single-label Learning Commands
- Multi-label Learning Commands
- Multi-label Learning Algorithms

Section 1) introduces seven scripts to cover each block of single-label learning flow introduced in [tutorial x](x) and the content is the same as section x in [blm manual](x).

Section 2) introduces two one-stop function scripts in blm-mll for multi-label learning flow, which help users conducting multi-label learning tasks with blm-mll. 

Section 3) describes the details of multi-label learning algorithms utilized in blm-mll to help users better understand the advantages of blm-mll system.



## Single-label Learning Commands

BioSeq-BLM is an updated system of BioSeq-BLM sharing the same single-label learning functions and command line tools. In this section, we move the command lines sections from blm manual to this documen. And group them as single-label learning commands serving as a comparison with multi-label learning commands in the next section. The intention of adding this section is for the scenario in which users deal with both single-label learning and multi-label learning tasks. For complete materials for these scripts, please refer to [manual](y).

These section introduces scripts about feature extraction, feature analysis and one-stop function for conduting single-label learning tasks using services provided by other scipts.



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

- `-ml {SVM,RF,CRF,CNN,LSTM,GRU,Transformer,WeightedTransformer}`

  The machine-learning algorithm for constructing predictor, for example: Support Vector Machine (SVM).

  Different from options in blm corresponding script, `Reformer` method is unavaliable in blm-mll.

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



## Multi-label Learning Commands

We add two scripts for applying multi-label learning tasks: BioSeq-BLM_Seq_mll.py and BioSeq-BLM_Res_mll.py. The structure of multi-label learning scripts is the same as that of single-label Learning scripts presented above. It’s worth noting that there exists some changes in the options of multi-label learning commands. In [tutorial ?](x), we demonstrate the relationship between blm-mll and blm and know that the multi-label learning flow shares manny underlying services with single-label learning flow like feature extraction and feature analysis. Due to the differences between multi-label learning and single-label Learning task, some function or methods provided by single-label Learning scripts are not avaliable in multi-label learning scripts, resulting in some differences. 

For more infomation of the multi-label learning algorithms used in blm-mll, we give a detailed algorithm-oriented introduction in the next section [Multi-label Learning Algorithms](# Multi-label Learning Algorithms).



### BioSeq-BLM_Seq_mll.py

#### Synopsis

“BioSeq-BLM_Seq_mll.py” is an executive Python script used for achieving the one-stop multi-label learning
function at sequence level.



#### Required Options

- `-category {DNA,RNA,Protein} `

  The category of input sequences.

- `-mode {OHE,BOW,TF-IDF,TR,WE,TM,SR}`

  The feature extraction mode for input sequence which analogies with NLP, for example: bag of words (BOW).

  Different from options in Single-label Learning scripts, `AF` mode and `Labeled-LDA` method of `TM` mode are unavaliable in blm-mll due to design.

- `-mll {BR,CC,LP,RakelD,RakelO,CLR,FP,RT,MLkNN,BRkNNaClassifier,BRkNNbClassifier,MLARAM}`

  The multi-label learning algorithom for conducting multi-label learning tasks, for example: Binary Relevance (BR). See [Multi-label Learning Algorithms](#Multi-label Learning Algorithms) for more information.

- `-ml {SVM,RF,CNN,LSTM,GRU,Transformer,WeightedTransformer}`

  The single-learning algorithm for constructing sub-predictors for the multi-label learning algorithom specified by `-mll` , for example: Support Vector Machine (SVM). This option is required only by multi-label learning algorithoms categoried as [Problem Transformation](#sequence-level problem). Pay attention to that different multi-label learning algorithom supports different set of sub-predictors, please refer to [multi-label learning algorithms in blm-mll](x). For more information about these single-learning algorithms, please refer to [blm manual](x). 

- `-seq_file SEQ_FILE`

  The input sequence file in FASTA format.

- `-label_file LABEL_FILE`

  The multi-label file corresponding to input sequence file in CSV format with a header. Each following line holds $q$ dimentional multi-label label of a specific sequence.



#### Optional Options

- `-h, --help`

  Show this help message and exit.

- `-words {Kmer,RevKmer,Mismatch,Subsequence,Top-NGram,DR,DT}`

  If you select mode in ['BOW', 'TF-IDF', 'TR', 'WE', 'TM'], you should select word for corresponding mode, for example Mismatch. Pay attention to that different category has different words, please reference to [manual](bliu?).

- `-method METHODIf `

  If you select mode in ['OHE', 'WE', 'TM', 'SR'], you should select method for corresponding mode, for example, select 'LDA' for 'TM' mode, select 'word2vec' for 'WE' mode and so on. For different category, the methods belong to 'OHE' and 'SR' mode is different, please reference to manual.

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

- `-grid [{0,1} [{0,1} ...]] `

  Grid=0 for rough grid search, grid=1 for meticulous grid search.

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

- `-dr{PCA,KernelPCA,TSVD,none} `

  Choose method for dimension reduction.

- `-np NP `

  The dimension of main component after dimension reduction.

- `-rdb {no,dr} `

  Reduce dimension by: 

  'no'---none; 

  'dr'--- apply dimension reduction to parameter selection procedure.

- `-cost [COST_LOW [COST_HIGH [STEP]]] `

  Regularization parameter of 'SVM'. 

  Produce a sequence of parameter from low (inclusive)
      to high (exclusive) by step(default is 1) for parameter selection. If high is None (the default), then low is produced only.

- `-gamma[GAMMA_LOW [GAMMA_HIGH [STEP]]] `

  Kernel coefficient for 'rbf' of 'SVM'.

  Produce a sequence of parameter from low (inclusive)
      to high (exclusive) by step(default is 1) for parameter selection. If high is None (the default), then low is produced only.

- `-tree [TREE_LOW [TREE_HIGH [STEP]]] `

  Produce a sequence of parameter from low (inclusive)
      to high (exclusive) by step(default is 1) for parameter selection. If high is None (the default), then low is produced only.

- `-lr LR `

  The value of learning rate for deep learning.

- `-epochs EPOCHS `

  The epoch number for deep model training.

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

- `-metric {Acc,F1} `

  The metric for parameter selection(default=F1)

- `-cv {5,10,j}`

  The cross validation mode. 5 or 10: 5-fold or 10-fold cross validation, j: (character 'j') jackknife cross validation.

- `-ind_seq_file IND_SEQ_FILE`

  The independent test dataset in FASTA format.

- `-ind_label_file IND_LABEL_FILE`

  The corresponding multi-label file of independent test dataset in CSV format.

- `-fixed_len FIXED_LEN`

  The length of sequence will be fixed via cutting orpadding. If you don't set value for 'fixed_len', it will be the maximum length of all input sequences.

- `-format {tab,svm,csv,tsv}`

  The output format (default=csv). 

  tab --Simple format, delimited by TAB. 

  svm --The libSVM training data format. 

  csv, tsv --The format that can be loaded into a spreadsheet program.

- `-bp {0,1}`

  Select use batch mode or not, the parameter will change the directory for generating file based on the method you choose.



> note
>
> compared with single-label learning flow
>
> remove blmx, feature selection, rdb, sampling 
>
> require mll and ml where ml means but not 



### BioSeq-BLM_Res_mll.py

#### Synopsis

“BioSeq-BLM_Res_mll.py” is an executive Python script used for achieving the one-stop multi-label learning
function at sequence level.



#### Required Options

- `-category {DNA,RNA,Protein} `

  The category of input sequences.

- `-method `

  Please select feature extraction method for residue level analysis.

- `-mll {BR,CC,LP,RakelD,RakelO,CLR,FP,RT,MLkNN,BRkNNaClassifier,BRkNNbClassifier,MLARAM}`

  The multi-label learning algorithom for conducting multi-label learning tasks, for example: Binary Relevance (BR). See [Multi-label Learning Algorithms](#Multi-label Learning Algorithms) for more information.

- `-ml {SVM,RF,CNN,LSTM,GRU,Transformer,WeightedTransformer}`

  The single-learning algorithm for constructing sub-predictors for the multi-label learning algorithom specified by `-mll` , for example: Support Vector Machine (SVM). This option is required only by multi-label learning algorithoms categoried as [Problem Transformation](#sequence-level problem). Pay attention to that different multi-label learning algorithom supports different set of sub-predictors, please refer to [multi-label learning algorithms in blm-mll](x). For more information about these single-learning algorithms, please refer to [blm manual](x). 

- `-window WINDOW`

  use sliding window technique to transform sequence-labelling question to classification question.

  The window size when construct sliding window technique for allocating every label a short sequence.

- `-seq_file SEQ_FILE`

  The input sequence file in FASTA format.

- `-label_file LABEL_FILE`

  The multi-label file corresponding to input sequence file in CSV format with a header. Each following line holds $L*q$ multi-label labels of a specific sequence, where $L$ is the number of residues in the sequence and $q$ is the dimension of multi-label of every residue.



#### Optional Options

- `-h, --help `

  Show this help message and exit.

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

- `-dr {PCA,KernelPCA,TSVD,none} `

  Choose method for dimension reduction.

- `-np NP `

  The dimension of main component after dimension reduction.

- `-rdb {no,dr} `

  Reduce dimension by: 

  'no'---none; 

  'dr'---apply dimension reduction to parameter selection procedure.

- `-grid [{0,1} [{0,1} ...]] `

  Grid=0 for rough grid search, grid=1 for meticulous grid search.

- `-cost [COST_LOW [COST_HIGH [STEP]]] `

  Regularization parameter of 'SVM'. 

  Produce a sequence of parameter from low (inclusive)
      to high (exclusive) by step(default is 1) for parameter selection. If high is None (the default), then low is produced only.

- `-gamma[GAMMA_LOW [GAMMA_HIGH [STEP]]] `

  Kernel coefficient for 'rbf' of 'SVM'.

  Produce a sequence of parameter from low (inclusive)
      to high (exclusive) by step(default is 1) for parameter selection. If high is None (the default), then low is produced only.

- `-tree [TREE_LOW [TREE_HIGH [STEP]]] `

  Produce a sequence of parameter from low (inclusive)
      to high (exclusive) by step(default is 1) for parameter selection. If high is None (the default), then low is produced only.

- `-lr LR `

  The value of learning rate for deep learning.

- `-epochs EPOCHS `

  The epoch number for deep model training.

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

- `-ind_seq_file IND_SEQ_FILE `

  The independent test dataset in FASTA format.

- `-ind_label_file IND_LABEL_FILE `

  The corresponding multi-label file of independent test dataset in CSV format.

- `-fixed_len FIXED_LEN`

  The length of sequence will be fixed via cutting orpadding. If you don't set value for 'fixed_len', it will be the maximum length of all input sequences.

- `-format {tab,svm,csv,tsv}`

  The output format (default=csv). tab –Simple format, delimited by TAB. svm --The libSVM training data format. csv, tsv --The format that can be loaded into a spreadsheet program.

- `-bp {0,1}`

  Select use batch mode or not, the parameter will change the directory for generating file based on the method you choose.



## Multi-label Learning Algorithms

In this section, we introduce command lines and details for each multi-label learning algorithms used in blm-mll for sake of using your multi-label learning tasks easier. For more infomation, see tutorial [multi-label learning algorithms in blm-mll](x). 



BibTeX of scikit-multilearn

```
@ARTICLE{2017arXiv170201460S,
  author = {{Szyma{\'n}ski}, P. and {Kajdanowicz}, T.},
  title = "{A scikit-based Python environment for performing multi-label classification}",
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {1702.01460},
  primaryClass = "cs.LG",
  keywords = {Computer Science - Learning, Computer Science - Mathematical Software},
  year = 2017,
  month = feb
}
```



### Binary

#### Binary Relevance

##### Synopsis

Transforms a multi-label classification problem with L labels into L single-label separate binary classification problems using the same base classifier provided in the constructor. The prediction output is the union of all per label classifiers



##### Options

- `-mll BR`

  set the multi-label learning algorithm as Binary Relevance.

- `-ml <method>`

  All blm predictors but `CRF` can serve as base single-label classifier of the selected multi-label classifier. Valid methods are listed here: `SVM`, `RF`, `CNN`, `LSTM`, `GRU`, `Transformer`, `Weighted-Transformer`, `Reformer`



#### Classifier Chain

##### Synopsis

Constructs a bayesian conditioned chain of per label classifiers

This class provides implementation of Jesse Read’s problem transformation method called Classifier Chains. For L labels it trains L classifiers ordered in a chain according to the Bayesian chain rule.

The first classifier is trained just on the input space, and then each next classifier is trained on the input space and all previous classifiers in the chain.

The default classifier chains follow the same ordering as provided in the training set, i.e. label in column 0, then 1, etc.



##### BibTeX

```tex
@inproceedings{read2009classifier,
  title={Classifier chains for multi-label classification},
  author={Read, Jesse and Pfahringer, Bernhard and Holmes, Geoff and Frank, Eibe},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={254--269},
  year={2009},
  organization={Springer}
}
```



##### Options

- `-mll CC`

  set the multi-label learning algorithm as Classifier Chain.

- `-ml <method>`



### Label Combination

#### Label PowerSet

##### Synopsis

Transform multi-label problem to a multi-class problem

Label Powerset is a problem transformation approach to multi-label classification that transforms a multi-label problem to a multi-class problem with 1 multi-class classifier trained on all unique label combinations found in the training data.

The method maps each combination to a unique combination id number, and performs multi-class classification using the classifier as multi-class classifier and combination ids as classes.



##### Options

- `-mll LP`

  set the multi-label learning algorithm as Label PowerSet.

- `-ml <method>`

  All blm predictors but `CRF` can serve as base single-label classifier of the selected multi-label classifier. Valid methods are listed here: `SVM`, `RF`, `CNN`, `LSTM`, `GRU`, `Transformer`, `Weighted-Transformer`, `Reformer`



### Pairwise & Threshold methods

Pairwise methods can work well, but they are very sensitive to the number of labels. One-vs-rest classifiers (e.g., `RT`) can be faster with large numbers of labels, but may not perform as well.

link到[weka](http://waikato.github.io/meka/methods/)，并指出如果要细致的调优参数，要用weka的GUI或。。。



#### Calibrated Label Ranking

##### Synopsis

This is an algorithom implemented by WEKA. For underlying implementation, please refer to [weka documentation](https://mulan.sourceforge.net/doc/).

For more information, see Fuernkranz, Johannes, Huellermeier, Eyke, Loza Mencia, Eneldo, Brinker, Klaus (2008). Multilabel classification via calibrated label ranking. Machine Learning. 73(2):133--153.



##### BibTeX

```tex
@article{Fuernkranz2008,
  author = {Fuernkranz, Johannes and Huellermeier, Eyke and Loza Mencia, Eneldo and Brinker, Klaus},
  journal = {Machine Learning},
  number = {2},
  pages = {133--153},
  title = {Multilabel classification via calibrated label ranking},
  volume = {73},
  year = {2008},
}
```

[ref](https://mulan.sourceforge.net/doc/)



##### Options

- `-mll CLR`

  set the multi-label learning algorithm as Calibrated Label Ranking.

- `-ml <method>`

  All blm predictors but `CRF` can serve as base single-label classifier of the selected multi-label classifier. Valid methods are listed here: `SVM`, `RF`, `CNN`, `LSTM`, `GRU`, `Transformer`, `Weighted-Transformer`, `Reformer`





#### Fourclass Pairwise

##### Synopsis

The Fourclass Pairwise (FW) method. Trains a multi-class base classifier for each pair of labels ($(L*(L-1))/2$ in total), each with four possible class values: {00,01,10,11} representing the possible combinations of relevant (1) /irrelevant (0) for the pair. Uses a voting + threshold scheme at testing time where e.g., 01 from pair jk gives one vote to label k; any label with votes above the threshold is considered relevant. 

This is an algorithom implemented by MEKA. For more information, please refer to [meka documentation](http://waikato.github.io/meka/meka.classifiers.multilabel.FW/#synopsis).



##### BibTeX

```tex
@article{MEKA,
  author = {Read, Jesse and Reutemann, Peter and Pfahringer, Bernhard and Holmes, Geoff},
  title = {{MEKA}: A Multi-label/Multi-target Extension to {Weka}},
  journal = {Journal of Machine Learning Research},
  year = {2016},
  volume = {17},
  number = {21},
  pages = {1--5},
  url = {http://jmlr.org/papers/v17/12-164.html},
}
```



##### Options

- `-mll FW`

- `-ml <method>`

  All blm predictors but `CRF` can serve as base single-label classifier of the selected multi-label classifier. Valid methods are listed here: `SVM`, `RF`, `CNN`, `LSTM`, `GRU`, `Transformer`, `Weighted-Transformer`, `Reformer`




#### Rank+Threshold

##### Synopsis

Duplicates multi-label examples into examples with one label each (one vs. rest). Trains a multi-class classifier, and uses a threshold to reconstitute a multi-label classification.

This is an algorithom implemented by MEKA. For more information, please refer to [meka documentation](http://waikato.github.io/meka/meka.classifiers.multilabel.FW/#synopsis).



##### BibTeX

```tex
@article{MEKA,
  author = {Read, Jesse and Reutemann, Peter and Pfahringer, Bernhard and Holmes, Geoff},
  title = {{MEKA}: A Multi-label/Multi-target Extension to {Weka}},
  journal = {Journal of Machine Learning Research},
  year = {2016},
  volume = {17},
  number = {21},
  pages = {1--5},
  url = {http://jmlr.org/papers/v17/12-164.html},
}
```



##### Options

- `-mll RT`

  set the multi-label learning algorithm as Rank + Threshold.

- `-ml <method>`

  All blm predictors but `CRF` can serve as base single-label classifier of the selected multi-label classifier. Valid methods are listed here: `SVM`, `RF`, `CNN`, `LSTM`, `GRU`, `Transformer`, `Weighted-Transformer`, `Reformer`



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

kNN classification method adapted for multi-label classification

MLkNN builds uses k-NearestNeighbors find nearest examples to a test class and uses Bayesian inference to select assigned labels.



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

```
@INPROCEEDINGS{7395756,
  author={F. Benites and E. Sapozhnikova},
  booktitle={2015 IEEE International Conference on Data Mining Workshop (ICDMW)},
  title={HARAM: A Hierarchical ARAM Neural Network for Large-Scale Text Classification},
  year={2015},
  volume={},
  number={},
  pages={847-854},
  doi={10.1109/ICDMW.2015.14},
  ISSN={2375-9259},
  month={Nov},
}
```



##### Options

- `-mll MLARAM`

  set the multi-label learning algorithm as MLARAM.

- `-mll_v <value> ` `--MLARAM_vigilance <value>` should be within [0, 1]

  parameter for adaptive resonance theory networks,  controls how large a hyperbox can be, 1 it is small (no compression), 0 should assume all range. Normally set between 0.8 and 0.999, it is dataset dependent. It is responsible for the creation of the prototypes, therefore training of the network 

- `-mll_t <value>` ` --MLARAM_threshold <value>`  should be within [0, 1)

  controls how many prototypes participate by the prediction, can be changed for the testing phase.
