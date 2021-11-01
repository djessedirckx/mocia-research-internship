#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
# your job code goes here, e.g.:
source mocia_env/bin/activate
python train_model.py
