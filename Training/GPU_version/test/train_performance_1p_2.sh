#!/bin/bash

################ introduction #################
# Welcome to the TinyBERT project
# The whole task-distill training includes two step.
# This is the second step: prediction layer distillation
# Before bashing the shell file, please change the current dir(sharing the same dir with the folder"test")

################ compiling ##################
CUDA_VISIBLE_DEVICES="6" \
python -m torch.distributed.launch --nproc_per_node 1 ./main.py \
	      --pred_distill  \
        --teacher_model ./bert_base_uncased_ft_sst \
        --student_model ./tmp_tinybert_dir \
        --data_dir ./glue_data/SST-2 \
        --task_name SST-2 \
        --output_dir ./TinyBERT_dir \
        --aug_train \
        --learning_rate 3e-5 \
        --num_train_epochs  3 \
        --eval_step 100 \
        --max_seq_length 64 \
        --train_batch_size 32 \
	      --do_lower_case \
	      --fps_acc_dir ./output \
	      --performance
