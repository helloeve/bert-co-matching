#!/bin/bash

export BERT_BASE_DIR=bert_base_uncased
export SQUAD_DIR=data/
export job=test_run
export savepath=models/${job}/

### Training
python run_squad.py \
  --do_train=True \
  --do_predict=False \
  --do_lower_case=True \
  --train_file $SQUAD_DIR/hotpot_train_v1.1.ordered_support_only.json \
  --predict_file $SQUAD_DIR/hotpot_dev_distractor_v1.ordered_support_only.json \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --init_checkpoint $BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ${savepath}

## Inference
python run_squad.py \
  --do_train=False \
  --do_predict=True \
  --do_lower_case=True \
  --train_file $SQUAD_DIR/hotpot_train_v1.1.ordered_support_only.json \
  --predict_file $SQUAD_DIR/hotpot_dev_distractor_v1.ordered_support_only.json \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --init_checkpoint $BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ${savepath}
