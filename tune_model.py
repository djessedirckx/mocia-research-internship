from typing import Dict, List

import keras_tuner as kt
import pandas as pd

from keras_tuner.tuners.randomsearch import RandomSearch
from sklearn.model_selection import train_test_split

from hyperparameter_tuning.MatchNetHyperModel import MatchNetHyperModel
from hyperparameter_tuning.RandomSearchConfig import RandomSearchConfig
from eda_preprocessing.DataPreprocessor import DataPreprocessor
from eda_preprocessing.DataCreator import DataCreator
from model.MatchNetConfig import MatchNetConfig

def random_search(matchnet_config: MatchNetConfig, search_config: RandomSearchConfig):
    input_file = 'tadpole_challenge/TADPOLE_D1_D2.csv'
    data_preprocessor = DataPreprocessor(input_file, label_forwarding=False)
    data_creator = DataCreator(window_length=search_config.cov_input_shape[0], prediction_horizon=matchnet_config.pred_horizon)

    study_df, missing_masks = data_preprocessor.preprocess_data()

    # Make label binary, mark all Dementia instances as positive
    study_df['DX'] = study_df['DX'].replace('Dementia', 1)
    study_df['DX'] = study_df['DX'].replace(
        ['MCI', 'NL', 'MCI to Dementia', 'NL to MCI', 'MCI to NL', 'Dementia to MCI', 'NL to Dementia'], 0)

    trajectory_labels, ptids = [], []
    for ptid, trajectory in study_df.groupby("PTID"):
        trajectory_labels.append(
            1) if 1 in trajectory['DX'].values else trajectory_labels.append(0)
        ptids.append(ptid)

    # Split data into train, val & test
    train_trajectories, test_trajectories, train_labels, test_labels = train_test_split(
        ptids, trajectory_labels, test_size=0.2, random_state=42, stratify=trajectory_labels)
    train_trajectories, val_trajectories = train_test_split(
        train_trajectories, test_size=0.25, random_state=42, stratify=train_labels)

    # Prepare training, validation & test data
    train_measurement_labels, train_true_labels, train_windows, train_masks = prepare_data(
        data_creator, study_df, missing_masks, train_trajectories)
    val_measurement_labels, val_true_labels, val_windows, val_masks = prepare_data(
        data_creator, study_df, missing_masks, val_trajectories)
    test_measurement_labels, test_true_labels, test_windows, test_masks = prepare_data(
        data_creator, study_df, missing_masks, test_trajectories)

    train_data = [train_windows, train_masks]
    train_labels = train_measurement_labels
    validation_data = [val_windows, val_masks]
    validation_labels = val_measurement_labels

    # Initialise search model
    search_model = MatchNetHyperModel(search_config, matchnet_config)

    tuner = RandomSearch(
        search_model,
        objective=kt.Objective('val_convergence_metric', direction='max'),
        max_trials=10,
        overwrite=True,
        directory='output',
        project_name='test',
        seed=42
    )

    tuner.search(x=train_data, y=train_labels, batch_size=128, epochs=50, sample_weight=train_true_labels, validation_data=(
        validation_data, validation_labels, val_true_labels), validation_batch_size=len(val_true_labels))
  
def prepare_data(data_creator: DataCreator, study_df: pd.DataFrame, missing_masks: pd.DataFrame, trajectories: List):
    windows = study_df.loc[study_df['PTID'].isin(trajectories)]
    masks = missing_masks.loc[missing_masks['PTID'].isin(trajectories)]
    return data_creator.create_data(windows, masks)    

if __name__ == '__main__':
    convergence_weights = [
        (1, 1),
        (1, 1),
        (1, 1),
    ]
    matchnet_config = MatchNetConfig(pred_horizon=1, convergence_weights=convergence_weights)
    search_config = RandomSearchConfig((3, 35), (3, 35))


    random_search(matchnet_config, search_config)

