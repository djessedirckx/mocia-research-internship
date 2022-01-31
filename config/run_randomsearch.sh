#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=16G

# Activate virtualenv and run randomsearch job
source env/bin/activate
python tune_model.py --prediction_horizon 2 --cross_val_splits 5 --max_trials 100 --label_forwarding --oversampling --weight_regularisation
