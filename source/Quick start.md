# Quickstart

This quickstart example shows how to easily 

**给出简单的例子，主要展示脚本使用方法，而不是论文里的case study**





## Collecting the data

给出数据获取方式，放到 data/ 目录下，确认格式

（数据选择小



## Conducting Multi-label Learning Tasks by k-fold Cross-validation

```bash
# BR RF
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll BR -ml RF -tree 3 -seq_file ../data/dev/seq_in_data.fasta -label_file ../data/dev/seq_in_label.csv -bp 1 -metric F1
```

result



## Conducting Multi-label Learning Tasks by Independent Testment

```bash
# BR RF
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll BR -ml RF -tree 3 -seq_file ../data/dev/seq_in_data.fasta -label_file ../data/dev/seq_in_label.csv -bp 1 -metric F1 -ind_seq_file ../data/dev/seq_in_data.fasta -ind_label_file ../data/dev/seq_in_label.csv
```

result



