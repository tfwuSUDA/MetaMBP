python ../main/Finetuning_DS_16.py \
 -expmode 1  -save_model 0 -timeID '' \
 -loss_fun "BCE" -use_meta 1 \
 -early_stop 0 -use_cuda 1 -No_ Finetuning_DS_16 -save_best 0 \
 -scheduler 0 -step_size 300 -gamma 0.5 -dropout 0.2 -learning_rate 1e-4 -weight_decay 5e-4 \
 -emb_dim 128 -epoch_num 1000 -batch_size 256 -seed 1200 -max_len 100 \
 -meta_model "../models/LSTM_attn_peptide_feature_encoder_32filters_5way_20shot.pkl" \
 -dataset "../data/task_data/Multi label Dataset/Multi-data-finetune/16class_hasdup_pos_neg.tsv"