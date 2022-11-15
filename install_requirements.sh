#!/bin/bash

conda install conda
conda install conda-build
conda install tqdm
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c huggingface transformers==4.14.1 tokenizers==0.10.3
conda install protobuf
pip install hydra-core --upgrade
pip install airdialogue-essentials
pip install tensorflow
pip install google-auth google-auth-oauthlib google-pasta 
pip install accelerate wandb flatten_dict 

conda develop "$PWD/offline_airdialogue"