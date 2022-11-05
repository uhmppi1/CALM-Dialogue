#!/bin/bash

python preprocess_instruction.py \
        --data_dir instruction_sample \
        --out_file ../../data/instruction_ad/train_instruction.json #\
#        --limit 100