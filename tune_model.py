import os

from datetime import datetime
from typing import Dict, List

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
from model.MatchNet import MatchNet
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

    # Obtain best model
    best_model = tuner.get_best_models()[0]

    # Evaluate model on test data
    window_length = best_model.layers[0].input_shape[0][1]
    data_creator = DataCreator(
        window_length=window_length, prediction_horizon=matchnet_config.pred_horizon)
    test_measurement_labels, test_true_labels, test_windows, test_masks = prepare_data(
        data_creator, study_df, missing_masks, test_trajectories)

    test_results=best_model.evaluate([test_windows, test_masks], test_measurement_labels, sample_weight = test_true_labels,
            batch_size = len(test_true_labels))

    tuner.results_summary(num_trials=10)

    # Store results
    save_statistics(matchnet_config, best_model, test_results, tuner.get_best_hyperparameters(num_trials=1)[0], now)

def prepare_data(data_creator: DataCreator, study_df: pd.DataFrame, missing_masks: pd.DataFrame, trajectories: List):
    windows=study_df.loc[study_df['PTID'].isin(trajectories)]
    masks=missing_masks.loc[missing_masks['PTID'].isin(trajectories)]
    return data_creator.create_data(windows, masks)

def save_statistics(config: MatchNetConfig, model: MatchNet, scores: List, hyperparameters: List, now: str):

    # Create output directory based on current date and time
    path = os.path.join(config.output_path, now)
    os.makedirs(path)

    # Save all statistics as binary numpy files
    test_scores = {
        'loss': scores[0],
        'au_roc': scores[3],
        'au_prc': scores[2],
        'convergence': scores[1]
    }
    np.save(f'{path}/test_scores', test_scores)
    np.save(f'{path}/test_hyperparams', hyperparameters)

    # Store model
    model.save(f'{path}/model.hdf5')

    print(f'Stored output in {path}, training finished')

if __name__ == '__main__':

    # TODO --> make dynamic
    convergence_weights=[
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1)
    ]
    matchnet_config= MatchNetConfig(
        pred_horizon = 1, convergence_weights = convergence_weights, val_frequency = 5,
        output_path='output/test_set')

    random_search(matchnet_config)
