import argparse

from datetime import datetime
from typing import List

import keras_tuner as kt
import numpy as np
import pandas as pd

from keras_tuner.oracles import RandomSearch
from sklearn.model_selection import train_test_split

from hyperparameter_tuning.MatchNetHyperModel import MatchNetHyperModel
from hyperparameter_tuning.MatchNetTuner import MatchNetTuner
from hyperparameter_tuning.RandomSearchConfig import RandomSearchConfig
from eda_preprocessing.DataPreprocessor import DataPreprocessor
from eda_preprocessing.DataCreator import DataCreator
from model.MatchNetConfig import MatchNetConfig


def random_search(matchnet_config: MatchNetConfig):

    # Load data and perform initial pre-processing
    input_file = 'tadpole_challenge/TADPOLE_D1_D2.csv'
    data_preprocessor = DataPreprocessor(input_file, label_forwarding=False)
    study_df, missing_masks = data_preprocessor.preprocess_data()

    # Current datetime to be able to properly distinguish between search outputs
    now = datetime.now().isoformat()

    print(f'\nExecuting randomsearch for prediction_horizon={matchnet_config.pred_horizon}\n')

    # Create configuration for the desired sliding window length
    search_config = RandomSearchConfig(now, matchnet_config.pred_horizon)

    # Make label binary, mark all Dementia instances as positive
    study_df['DX'] = study_df['DX'].replace('Dementia', 1)
    study_df['DX'] = study_df['DX'].replace(
        ['MCI', 'NL', 'MCI to Dementia', 'NL to MCI', 'MCI to NL', 'Dementia to MCI', 'NL to Dementia'], 0)

    trajectory_labels, ptids = [], []
    for ptid, trajectory in study_df.groupby("PTID"):
        trajectory_labels.append(
            1) if 1 in trajectory['DX'].values else trajectory_labels.append(0)
        ptids.append(ptid)

    # Split data into train/validation & test
    train_trajectories, test_trajectories, train_labels, test_labels = train_test_split(
        ptids, trajectory_labels, test_size=0.2, random_state=42, stratify=trajectory_labels)

    # Configure search
    tuner_oracle = RandomSearch(
        objective=kt.Objective('val_convergence_metric', direction='max'),
        max_trials=100, # TODO --> make hyperparameter
        seed=42
    )

    search_model = MatchNetHyperModel(search_config, matchnet_config)
    tuner = MatchNetTuner(
        oracle=tuner_oracle,
        hypermodel=search_model,
        prediction_horizon=matchnet_config.pred_horizon,
        directory='output/random_search',
        project_name=search_config.output_folder)

    # Execute cross-validated random hyperparameter search
    tuner.search(trajectories=np.array(train_trajectories), trajectory_labels=train_labels, study_df=study_df, missing_masks=missing_masks)

    # Show hyperparameters of 10 best trials
    tuner.results_summary(num_trials=10)

def prepare_data(data_creator: DataCreator, study_df: pd.DataFrame, missing_masks: pd.DataFrame, trajectories: List):
    windows=study_df.loc[study_df['PTID'].isin(trajectories)]
    masks=missing_masks.loc[missing_masks['PTID'].isin(trajectories)]
    return data_creator.create_data(windows, masks)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tune MATCH-net hyperparameters for a desired prediction horizon')
    parser.add_argument('--prediction_horizon', type=int, help='Number of events in the future to predict')
    args = parser.parse_args()

    # TODO --> make dynamic
    convergence_weights=[
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1)
    ]
    matchnet_config= MatchNetConfig(
        pred_horizon = args.prediction_horizon, convergence_weights = convergence_weights,
        output_path='output/test_set')

    random_search(matchnet_config)
