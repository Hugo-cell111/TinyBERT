#!/bin/bash

################ introduction #################
# Welcome to the TinyBERT project
# The whole task-distill training includes two step.
# This is the first step: intermediate layer distillation
# Before bashing the shell file, please change the current dir(sharing the same dir with the folder"test")

################ compiling ##################
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
python -m torch.distributed.launch --nproc_per_node 8 main.py \
             --teacher_model ./bert_base_uncased_ft_sst \
             --student_model ./General_TinyBERT_4L_312D \
             --data_dir ./glue_dir/SST-2 \
             --task_name SST-2 \
             --output_dir ./tmp_tinybert_dir \
             --max_seq_length 64 \
             --train_batch_size 32 \
             --num_train_epochs 10 \
             --aug_train \
             --do_lower_case \
             --fps_acc_dir ./output \
             --performance