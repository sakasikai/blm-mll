# Tutorial

## Introduction

> focus on multi-label learning tasks in bioinfomatics

With the rapid development of biological sequencing, Genome and Protein sequences are growing rapidly but their structure and function and are still unknown. 

Compared with traditional lab-based methods, bioinfomatics plays an important role in exploring the relationship barried in the huge amont of biological sequences in various databases. 

Many statistical approach of pattern recognition and deep learning technologies have been applied in the filed of bioinformatics and achive great success. However many of them are restricted by the classical *only-one-label-per-pattern* supervised learning paradigm (also known as single-label learning, SLL) and fail to satisfy that requires mutiple outputs against a pattern(aka multi-label learning, MLL).  

It’s worth noting that many biological tasks comply MLL paradigm such as gene function prediction, protein function prediction, proteins 3D structure prediction and protein subcellular multilocation prediction (proteins may simultaneously exist at, or move between, two or more different subcellular locations).

> research niche => contribution => meanings

Dispite many works have promoted this very important field in bioinformatcis, they sitll suffer from some problems such as the ignorance of relationships between labels and the limited performance. Moverover,  and lack of generalization of the complicated flow of MLL.

In this work, we propose a lightweight multi-label learnig framework to help researchers quickly find candidates of MLL problem transformation strategy combined with base SLL classifiers to apply to their specific biological MLL problems.



To make study of MLL in bioinformatcis easily and efficiently, we develop a system called 😾 to implement the MLL framework which also provide a automatic flow including representation, modeling, training, testment and evaluation.



The experiments on seven datasets have proved that the system achives highly comparable or even superior results than that  of existing state-of-the-art works, which means our system is a powerful tool capable of improving performance or serving as benchmark for extensive MLL applications in bioinformatcis. 

Instead of complicated customization workflow for a specific MLL problem, our work raise a general framework and provide systemetic approaches for studying both sequence-level and residue-level MLL problems in bioinformatcis from different strategies. This means our system can be applied in many different MLL tasks in bioinformatcis and inspire more powerful customized methods, which is expected to give a widespread influence on this very important field.

It is worth noting that the data representation module in our system meaning are supported by BioSeq-BLM which constructs a BLM against biological sequeces while gets restricted in SLL paradigm due to the complicated modeling and training methods of MLL paradigm. From this point of view, our automatic system extending the BLM into an MLL flow is of great significance for better application of the BLM in bioinformatcis, bringing new technologies and powerful reference for the MLL study in bioinformatcis.



> navigation of chapters

For the working flow of system, please refer to Architecture and Pipeline

For the validation of system, please refer to Validation

For the manual of the system, please refer to Command line tools

For the quick start of the system, please refer to Quick start

For the access of the stand-alone package of the system, please refer to Installation Download

Additional Information (how to cite)





## Architecture

We propose a general MLL strategy framework, which can deal with both sequence-level and residue-level MLL problem. 

For sequence-level problem, the strategies can be categrized into two groups, i.e. Problem Transformation and Algorithm Adaptation. 



### sequence-level problem

#### Problem Transformation methods

```reStructuredText
+----------------------+--------------------------+-------------+
|      mll taxonomy    |       mll algorithm      | base method |
+======================+==========================+=============+
|                      |   Binary Relevance(BR)   |   ML + dl   |
|       Binary         +--------------------------+-------------+
|                      |   Classifier Chains(CC)  |     ml      |
+----------------------+--------------------------+-------------+
|   Label Combination  |    Label Powerset (LP)   |   ml + dl   |
+----------------------+--------------------------+-------------+
|                      | Calibrated Label Ranking |             |
|                      +--------------------------+             |
| Pairwise & Threshold |    Fourclass Pairwise    |   ml + dl   |
|                      +--------------------------+             |
|                      |     Rank   Threshold     |             |
+----------------------+--------------------------+-------------+
|                      |      RakelD              |             |
|   Ensembles of MLL   +--------------------------+   ml + dl   |
|                      |      RakelO              |             |
+----------------------+--------------------------+-------------+  
```



abbreviation in table

- mll means multi-lablel learning
- ml means machine learning methods in blm-mll: SVM, RF

- dl means deep learning methods in blm-mll: CNN, LSTM, GRU, Transformer, Weighted-transformer

#### Algorithm Adaptation methods

```reStructuredText
+--------------------+--------------------------+-------------+
|     mll taxonomy   |             mll algorithm              |
+====================+==========================+=============+
|       by kNN       |                ml_kNN                  |
|                    +----------------------------------------+
|                    |           BRkNNaClassifier             |
|                    +----------------------------------------+
|                    |           BRkNNbClassifier             |
+--------------------+----------------------------------------+
| by neural networks |       MLARAM(Multi-label ARAM)         |
+--------------------+----------------------------------------+ 
```



### residue-level problem

just like blm do, we use a sliding window strategy in blm-mll to transform residue-level problems into sequence-level problems which means all of the N mll algorithms can be applied to residue-level problems.







## Pipeline

Given the sequence data and labels for a specific biological MLL task, our system will automatically extract features, construct the compound predictor in case of the strategy, evaluate the performance, and analyze the results.



图！



## Validation





## Command line tools

介绍如何使用，具体 导航到 单独页







