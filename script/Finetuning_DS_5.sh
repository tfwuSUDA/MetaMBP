python ../main/Finetuning_DS_5.py \
 -expmode 1  -save_model 1000 -timeID '' \
 -loss_fun "Focal" -use_meta 1 -class_num 5 \
 -early_stop 0 -use_cuda 1 -No_ Finetuning_DS_5 -save_best 0 \
 -scheduler 0 -step_size 300 -gamma 0.5 -dropout 0.2 -learning_rate 1e-4 -weight_decay 5e-4 \
 -emb_dim 128 -epoch_num 1000 -batch_size 64 -seed 0 -max_len 100 \
 -mata_model "../models/LSTM_attn_peptide_feature_encoder_32filters_5way_20shot.pkl" \
 -dataset "../data/MLBP"