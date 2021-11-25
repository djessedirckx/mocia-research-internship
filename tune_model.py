import argparse

from datetime import datetime
from typing import List

import keras_tuner as kt
import numpy as np
import pandas as pd

from keras_tuner.oracles import RandomSearch
from sklearn.model_selection import StratifiedKFold

from hyperparameter_tuning.MatchNetHyperModel import MatchNetHyperModel
from hyperparameter_tuning.MatchNetTuner import MatchNetTuner
from hyperparameter_tuning.RandomSearchConfig import RandomSearchConfig
from eda_preprocessing.DataPreprocessor import DataPreprocessor
from eda_preprocessing.DataCreator import DataCreator
from model.config.MatchNetConfig import MatchNetConfig


def random_search(matchnet_config: MatchNetConfig, n_splits: int = 5, max_trials: int = 100):

    # Load data and perform initial pre-processing
    input_file = 'tadpole_challenge/TADPOLE_D1_D2.csv'
    data_preprocessor = DataPreprocessor(input_file, label_forwarding=matchnet_config.label_forwarding)
    study_df, missing_masks, forwarded_indexes = data_preprocessor.preprocess_data()

    # Current datetime to be able to properly distinguish between search outputs
    now = datetime.now().isoformat()

    print(f'\nExecuting randomsearch for prediction_horizon={matchnet_config.pred_horizon}\n')

    # Make label binary, mark all Dementia instances as positive
    study_df['DX'] = study_df['DX'].replace('Dementia', 1)
    study_df['DX'] = study_df['DX'].replace(
        ['MCI', 'NL', 'MCI to Dementia', 'NL to MCI', 'MCI to NL', 'Dementia to MCI', 'NL to Dementia'], 0)

    trajectory_labels, ptids = [], []
    for ptid, trajectory in study_df.groupby("PTID"):
        trajectory_labels.append(
            1) if 1 in trajectory['DX'].values else trajectory_labels.append(0)
        ptids.append(ptid)

    trajectory_labels = np.array(trajectory_labels)
    ptids = np.array(ptids)

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    test_au_rocs = np.zeros(n_splits)
    test_au_prcs = np.zeros(n_splits)
    all_val_au_roc = np.zeros((n_splits, max_trials))
    all_val_au_prc = np.zeros((n_splits, max_trials))
    best_params = []
    for cross_run, (train_idx, test_idx) in enumerate(kfold.split(ptids, trajectory_labels)):
        train_trajectories = ptids[train_idx]
        train_trajectory_labels = trajectory_labels[train_idx]
        test_trajectories = ptids[test_idx]

        val_au_rocs = np.zeros(max_trials)
        val_au_prcs = np.zeros(max_trials)

        # Configure search
        tuner_oracle = RandomSearch(
            objective=kt.Objective('val_convergence_metric', direction='max'),
            max_trials=max_trials,
            seed=42
        )

        # Create configuration for the desired sliding window length
        search_config = RandomSearchConfig(now, matchnet_config.pred_horizon, cross_run)
        search_model = MatchNetHyperModel(search_config, matchnet_config)
        tuner = MatchNetTuner(
            oracle=tuner_oracle,
            hypermodel=search_model,
            prediction_horizon=matchnet_config.pred_horizon,
            directory='/ceph/csedu-scratch/project/ddirckx/random_search', # TODO --> read from config
            project_name=search_config.output_folder)

        # Execute cross-validated random hyperparameter search
        tuner.search(
            trajectories=train_trajectories, 
            trajectory_labels=train_trajectory_labels, 
            study_df=study_df, 
            missing_masks=missing_masks, 
            forwarded_indexes=forwarded_indexes,
            oversampling=matchnet_config.oversampling)

        # Collect validation scores and compute standard deviations and means
        for i, trial in enumerate(tuner.oracle.trials.values()):
            val_au_rocs[i] = trial.metrics.metrics['val_au_roc']._observations[0].value[0]
            val_au_prcs[i] = trial.metrics.metrics['val_au_prc']._observations[0].value[0]

        all_val_au_roc[cross_run] = val_au_rocs
        all_val_au_prc[cross_run] = val_au_prcs

        # Prepare the test data
        best_model = tuner.get_best_models()[0]
        window_length = best_model.layers[0].input_shape[0][1]
        data_creator = DataCreator(window_length, matchnet_config.pred_horizon)
        test_measurement_labels, test_true_labels, _, test_metric_labels, test_windows, test_masks = prepare_data(data_creator, study_df, missing_masks, test_trajectories, forwarded_indexes)

        # Evaluate best model on test data
        evaluation_results = best_model.evaluate([test_windows, test_masks], test_measurement_labels, sample_weight=[test_true_labels, test_metric_labels], batch_size=len(test_true_labels))
        test_au_rocs[cross_run] = evaluation_results[3]
        test_au_prcs[cross_run] = evaluation_results[2]

        # Store hyperparameters of best trial
        best_params.append(tuner.get_best_hyperparameters(1)[0].values)
    
    print('\nCross validation finished, results on test data:')
    print(f'AUROC: {np.mean(test_au_rocs):.3f} - std={np.std(test_au_rocs):.3f}')
    print(f'AUPRC: {np.mean(test_au_prcs):.3f} - std={np.std(test_au_prcs):.3f}\n')
    print('----- Validation data metrics -----')

    for auroc, auprc in zip(all_val_au_roc, all_val_au_prc):
        print(f'AUROC: Std: {np.std(auroc)} - Mean: {np.mean(auroc)} -- AUPRC: Std: {np.std(auprc)} - Mean: {np.mean(auprc)}')

    print(f'Complete AUROC val stats: Std: {np.std(all_val_au_roc.ravel())} - Mean: {np.mean(all_val_au_roc.ravel())}')
    print(f'Complete AUPRC val stats: Std: {np.std(all_val_au_prc.ravel())} - Mean: {np.mean(all_val_au_prc.ravel())}')

    # Print best hyperparameters
    for i, param_set in enumerate(best_params):
        print(f'Best hyperparameters for partition: {i+1} - {param_set}\n')

def prepare_data(data_creator: DataCreator, study_df: pd.DataFrame, missing_masks: pd.DataFrame, trajectories: List, forwarded_indexes: List):
    windows=study_df.loc[study_df['PTID'].isin(trajectories)]
    masks=missing_masks.loc[missing_masks['PTID'].isin(trajectories)]
    return data_creator.create_data(windows, masks, forwarded_indexes)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tune MATCH-net hyperparameters for a desired prediction horizon')
    parser.add_argument('--prediction_horizon', type=int, help='Number of events in the future to predict')
    parser.add_argument('--cross_val_splits', type=int, help='Number of cross validation splits to use', default=5)
    parser.add_argument('--max_trials', type=int, help='Max number of trials to perform randomsearch')
    parser.add_argument('--label_forwarding', action='store_true', help='Employ label forwarding to passively increase amount of positive labels')
    parser.add_argument('--weight_regularisation', action='store_true', help='Use weight regularisation in model')
    parser.add_argument('--oversampling', action='store_true', help='Apply oversampling of the minority class in the training data')
    args = parser.parse_args()

    matchnet_config= MatchNetConfig(
        cov_input_features=35,
        mask_input_features=35,
        pred_horizon = args.prediction_horizon,
        output_path='/ceph/csedu-scratch/project/ddirckx/random_search',
        label_fowarding=args.label_forwarding,
        weight_regularisation=args.weight_regularisation,
        oversampling=args.oversampling)

    random_search(matchnet_config, n_splits=args.cross_val_splits, max_trials=args.max_trials)
