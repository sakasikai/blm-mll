# Quickstart

This quickstart example shows how to easily analysise

**简单的例子，主要展示脚本使用方法，不是论文里的case study**





```bash
# BR SVM
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll BR -ml SVM -cost 2 -gamma 3 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# BR, cnn
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll BR -ml CNN -epochs 2 -out_channels 50 -kernel_size 3 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -batch_size 30 -lr 1e-3 -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

```



