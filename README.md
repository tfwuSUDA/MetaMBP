# MetaMBP

This is the implementation of *MetaMBP: Few-shot multi-label prediction of bioactive peptides*

*based on deep metric meta-learning*

## Meta-learning Stage

Execute 

```bash
cd script
sh Meta_train.sh
```

for meta-training.

Execute 

```bash
cd script
sh Meta_test.sh
```

for meta-testing.

## Fine-tuning Stage

Execute 

```bash
cd script
sh Finetuning_DS_16.sh
```

for DS_16 dataset finetuning.

Execute 

```bash
cd script
sh Finetuning_DS_5.sh
```

for DS_5 dataset finetuning.

 