#!/bin/bash

################ introduction #################
# Welcome to the TinyBERT project
# This is the final step: evaluation!

################ parser setting ##################
# npu device id(using command npu-smi info -l to get the device id and set it as you want)
device_id='0,1,2,3,4,5,6,7'

################ compiling ##################
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
python -m torch.distributed.launch --nproc_per_node 8 main.py \
--do_eval \
--student_model ./TinyBERT_dir \
--data_dir ./glue_dir/SST-2 \
--task_name SST-2 \
--output_dir ./result_dir \
--do_lower_case \
--max_seq_length 64 \
--eval_batch_size 32