#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 \
python main.py \
--data_dir ./data/ \
--embedding_dim 300 \
--margin_value 1 \
--batch_size 100 \
--learning_rate 0.001 \
--n_generator 24 \
--n_rank_calculator 24 \
--eval_freq 10000000 \
--max_epoch 50