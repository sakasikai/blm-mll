# Quick start

This page give an example of runing our system to cnoduct biological multi-label tasks to help users get started quickly. 

First of all, confirm you have completed the installation of environment mentioned in [Installation Guide](https://blm-mll.readthedocs.io/en/latest/InstallationGuide.html) successfully. Then prepare the multi-data in `fasta` format and it’s corresponding label in `csv` format. Finally run commands describled in [Multi-label Learning Algorithms](https://blm-mll.readthedocs.io/en/latest/CommandLineTools.html#multi-label-learning-algorithms) to conduct multi-lable learning task with the dataset.



## Collecting multi-label data

Here we choose the task of identifying subcellular localization of non-coding RNA for our example. We download the dataset from [work](https://github.com/guofei-tju/ Identify-NcRNA-Sub-Loc-MKGHkNN) [[1]](#ref1), and prepare the directory structure as following:

```apl
.
code/
├── BioSeq-BLM_Seq_mllearn.py
└── ...
data/
├── miRNA.fasta
└── miRNA_label.csv
```

Run `head miRNA.fasta` for a glimpse of data in fasta format:

```apl
>transcript_id|MIMAT0014285 aae-miR-2765 Aedes aegypti miRNA
UGGUAACUCCACCACCGUUGGC
>transcript_id|MIMAT0014230 aae-miR-306-3p Aedes aegypti miRNA
GAGAGCACCUCGGUAUCUAAGC
>transcript_id|MIMAT0002178 hsa-miR-487a-3p Homo sapiens miRNA
AAUCAUACAGGGACAUCCAGUU
>transcript_id|MIMAT0004918 hsa-miR-892b Homo sapiens miRNA
CACUGGCUCCUUUCUGGGUAGA
>transcript_id|MI0010680 ssc-mir-124a-1 Sus scrofa miRNA
UCAAGACCAGACUCUGCCCUCCGUGUUCACAGCGGACCUUGAUUUAAUGUCAUACAAUUAAGGCACGCGGUGAAUGCCAAGAGCGGAGCCUACGGCUGCACUUGA
```

Run `head miRNA_label.csv` for a glimpse of label in csv format:

```apl
Exosome,Nucleus,Cytoplasm,Mitochondrion,Circulating,Microvesicle
0,1,1,0,0,0
0,1,1,0,0,0
1,0,0,0,1,0
1,0,0,0,0,1
1,0,0,0,0,0
1,0,0,1,0,1
1,0,0,0,0,0
0,1,1,0,0,0
1,0,0,0,0,1
```



## Conducting multi-label learning tasks with k-fold cross-validation

You can pick up any of the following commands to run with the dataset, each of which corresponds to a different combination of multi-label algorithm and data representation.

```bash
# BR algorithm with RF as base method, BOW(Kmer) as BLM
python BioSeq-BLM_Seq_mllearn.py -category RNA -mode BOW -words Kmer -mll BR -ml RF -tree 2 -seq_file ../data/miRNA.fasta -label_file ../data/miRNA_label.csv -bp 1 -metric F1 -mix_mode as_rna

# LP algorithm with RF as base method, BOW(Subsequence) as BLM
python BioSeq-BLM_Seq_mllearn.py -category RNA -mode BOW -words Subsequence -mll LP -ml RF -tree 2 -seq_file ../data/miRNA.fasta -label_file ../data/miRNA_label.csv -bp 1 -metric F1 -mix_mode as_rna

# RAkELd algorithm with LP(CNN) as base predictors, One-hot encoding as BLM
python BioSeq-BLM_Seq_mllearn.py -category RNA -mode OHE -method One-hot -mll RAkELd --RAkEL_labelset_size 3 -ml CNN -epochs 2 -out_channels 50 -kernel_size 3 -seq_file ../data/miRNA.fasta -label_file ../data/miRNA_label.csv -bp 1 -metric F1 -batch_size 30 -lr 1e-3 -fixed_len 500 -mix_mode as_rna

# RAkELo algorithm with LP(RF) as base predictors, TF-IDF(Kmer) as BLM
python BioSeq-BLM_Seq_mllearn.py -category RNA -mode TF-IDF -words Kmer -mll RAkELo -ml RF -tree 2 --RAkEL_labelset_size 3 --RAkELo_model_count 6 -seq_file ../data/miRNA.fasta -label_file ../data/miRNA_label.csv -bp 1 -metric F1 -mix_mode as_rna

# LP algorithm with SVM as base method, TopicModel LDA(BOW Kmer) as BLM
python BioSeq-BLM_Seq_mllearn.py -category RNA -mode TM -method LDA -in_tm BOW -words Kmer -mll LP -ml SVM -cost 2 -gamma 3 -seq_file ../data/miRNA.fasta -label_file ../data/miRNA_label.csv -bp 1 -metric F1 -mix_mode as_rna
```

Then you will see the console for leaning and testing information and find prediction summary under the results directory as follow. 

```apl
../results/batch/Seq/RNA/
├── BOW
│   ├── Kmer
│   │   ├── all_seq_file.fasta
│   │   ├── br_rf_tree_2.model
│   │   ├── cv_features[mll]_csv.txt
│   │   ├── final_results.txt
│   │   ├── opt_cv_features[mll]_csv.txt
│   │   ├── Opt_params.txt
│   │   └── prob_out.csv
│   └── Subsequence
│       ├── all_seq_file.fasta
│       ├── cv_features[mll]_csv.txt
│       ├── final_results.txt
│       ├── lp_rf_tree_2.model
│       ├── opt_cv_features[mll]_csv.txt
│       ├── Opt_params.txt
│       └── prob_out.csv
├── OHE
│   ├── One-hot
│   │   ├── all_seq_file.fasta
│   │   ├── cv_dl_features[mll]_.txt
│   │   ├── final_results.txt
│   │   └── prob_out.csv
│   └── PSSM
│       └── all_seq_file.fasta
├── TF-IDF
│   └── Kmer
│       ├── all_seq_file.fasta
│       ├── cv_features[mll]_csv.txt
│       ├── final_results.txt
│       ├── opt_cv_features[mll]_csv.txt
│       ├── Opt_params.txt
│       ├── prob_out.csv
│       └── rakelo_rf_RAkEL_labelset_size_3_RAkELo_model_count_6_tree_2.model
└── TM
    └── LDA
        └── BOW
            └── Kmer
                ├── all_seq_file.fasta
                ├── cv_features[mll]_csv.txt
                ├── final_results.txt
                ├── lp_svm_gamma_3_cost_2.model
                ├── opt_cv_features[mll]_csv.txt
                ├── Opt_params.txt
                └── prob_out.csv
```

The summary includes the optimal selected model, feature vectors extracted by BLM descriptor, final result of the multi-label task, and the optimal parameters corresponding to it.



(ref1)=

[1] Identify ncRNA subcellular localization via graph regularized k-local hyperplane distance nearest neighbor model on multi-kernel learning
