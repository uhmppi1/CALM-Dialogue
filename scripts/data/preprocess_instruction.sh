#!/bin/bash

python preprocess_instruction.py \
        --data_dir instruction_sample_train \
        --out_file ../../data/instruction_ad/train_instruction.json

python preprocess_instruction.py \
        --data_dir instruction_sample_dev \
        --out_file ../../data/instruction_ad/dev_instruction.json