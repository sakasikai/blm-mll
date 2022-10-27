# Tutorial

> 概括介绍本项目，从 blm 说起

In order to uncover the meanings of “book of life”, 155 different biological language models (BLMs) for DNA, RNA and protein sequence analysis are introduced and discussed in [our previous study]() , which are able to extract the linguistic properties of “book of life”. We extend the BLMs into a system called BioSeq-BLM for automatically representing and analyzing the sequence. 

> research niche

Dispite the powerful analyiing capability for biological sequences, BLM fails dealing with multi-label learning problems for its one lablel learning assumption.

> 介绍本系统

In this study, we upgrade system BioSeq-BLM to system BioSeq-BLM-Mll（名字保留修改） which utilizes multiple multi-label learning strategies and methods providied by BLM to deal with biological multi-label learning tasks.

> 从贡献上，说本系统和BLM的变化

Without changing the shared functions in BLM, BLM-Mll brings a powerful multi-label learning module into BLM system which gives a one-stop process for multi-label learning researchers . Futhurmore, BLM-Mll can use nearly all single-label learning predictors in BLMs to serve as base-methods of a multi-label learning algorithm.



> 介绍tutorial的结构，分几部分，每部分主要说什么

This tutorial can be split into x parts

part 1 give you an intro to multi-label learning task in bioinfo

part 2 will tell you how BLM-MLL tackle multi-label learning task

part 3 help you learn how to deal with your multi-label learning task by BLM-MLL



## multi-label learning tasks

> 一般性定义，研究的意义

Multi-label learning (MLL) is a supervised learning paradigm where each real-world entity is associated with a set of labels simultaneously. During the past decade, signifificant amount of progresses have been made towards this recent learning paradigm for its potention in improving performance of problems where a pattern may have more than one associated class.



> 缩小到生物学中，提出面临的挑战，

In bio-informatics domain, there are many important multi-label learning tasks for biological analysis , such as RNA-associated subcellular localizations, protein subcellular localization, … , etc.  



In this study, we divide multi-label learning tasks of bio-informatics domain into two class:

- Sequenece-level multi-label learning tasks
- Residue-level multi-label learning tasks



### Sequenece-level multi-label learning tasks

> 研究的任务



### Residue-level multi-label learning tasks

> 研究的任务



## blm

（从blm介绍起，然后说清楚本项目和blm的关系，然后介绍本项目的特点（主要是多标记学习任务）和贡献

介绍blm，能做什么，意义，

指出在多标记学习上的缺陷（nitche），引出做出本系统的思想

（指引blm的使用，然后说明本项目只使用增量



## blm-mll

介绍blm-mll的功能，说出意义，引出和blm的关系，

指明和blm的区别后，也指明本文档和blm旧文档之间的关系，

进一步，给出作者使用本文档，本系统的最佳方式。



## similarities and differences

(between blm and blm-mll

说明和blm的异同

- blm能做的，blm-mll都能做，且数据和命令行完全一样

- 输入数据不同，类型，格式

- 输出

- 相似度

- feature analysis

  dimension reduction

- evaluation



## How to use this document







## multi-label learning algorithms in blm-mll

涉及到blm方法作为基方法，要清晰指出不同之处（哪些是blm的，哪些是blm-mll的）

|      |      |
| ---- | ---- |
|      |      |
|      |      |
|      |      |



## Conducting multi-label learning tasks with blm-mll

介绍 scripts 和 cmd

(Furthermore, use cases are provided in Quick Start

### Example 1 RNA-associated subcellular localizations

命令行，参数，数据

结果评估



### Example 2 protein subcellular localization





### Example 3 



## Evaluation and interpsaretation of the blm-mll

metrics

feature analysis



### 







