#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=16G

# Activate virtualenv and run randomsearch job
source mocia_env/bin/activate
python tune_model.py --prediction_horizon 1 
