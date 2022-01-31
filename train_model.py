import argparse
from itertools import product
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from analysis.test_metrics import compute_c_index_score, compute_calibration_curve, compute_brier_score
from eda_preprocessing.DataPreprocessor import DataPreprocessor
from eda_preprocessing.DataCreator import DataCreator
from model.config.MatchNetConfig import MatchNetConfig
from model.model_builder import build_model

def train_model(matchnet_config: MatchNetConfig, n_splits: int = 5, max_epochs: int = 50, batch_size: int = 128, eval_time=0):
    # Load data and perform initial pre-processing
    input_file = 'tadpole_challenge/TADPOLE_D1_D2.csv'
    data_preprocessor = DataPreprocessor(input_file, label_forwarding=matchnet_config.label_forwarding)
    study_df, missing_masks, forwarded_indexes = data_preprocessor.preprocess_data()

    # Make label binary, mark all Dementia instances as positive
    study_df['DX'] = study_df['DX'].replace('Dementia', 1)
    study_df['DX'] = study_df['DX'].replace(
        ['MCI', 'NL', 'MCI to Dementia', 'NL to MCI', 'MCI to NL', 'Dementia to MCI', 'NL to Dementia'], 0)

    trajectory_labels, ptids, traj_lengths = [], [], []
    for ptid, trajectory in study_df.groupby("PTID"):
        labels = trajectory['DX'].values
        trajectory_labels.append(
            1) if 1 in labels else trajectory_labels.append(0)
        traj_lengths.append(np.where(labels == 1)[0][0] + 1 if 1 in labels else np.where(labels == 0)[0][-1] + 1)
        ptids.append(ptid)

    median_traj_length = np.median(traj_lengths)
    trajectory_labels = np.array(trajectory_labels)
    ptids = np.array(ptids)
    
    # Get all possible l1, l2 combinations
    l1 = [0.03]
    l2 = [0.001]
    combs = list(product(l1, l2))

    test_auroc = np.zeros(len(combs))
    test_auprc = np.zeros(len(combs))
    test_conv = np.zeros(len(combs))
    test_c_idx = np.zeros(len(combs))
    test_auroc_std = np.zeros(len(combs))
    test_auprc_std = np.zeros(len(combs))
    test_brier_score = np.zeros(len(combs))
    test_brier_std = np.zeros(len(combs))

    test_true_curves = []
    test_pred_curves = []
    
    for i, l1_l2 in enumerate(combs):        
        # Pass configuration to model config
        matchnet_config.l1 = l1_l2[0]
        matchnet_config.l2 = l1_l2[1]

        early_stopping=EarlyStopping(monitor = 'val_convergence_metric', patience = matchnet_config.val_frequency, verbose = 1, mode = 'max')
        data_creator = DataCreator(matchnet_config.window_length, matchnet_config.pred_horizon)

        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_au_rocs = np.zeros(n_splits)
        fold_au_prcs = np.zeros(n_splits)
        fold_conv = np.zeros(n_splits)
        fold_c_index = np.zeros(n_splits)
        fold_brier_score = np.zeros(n_splits)
        true_curves = []
        pred_curves = []
        for cross_run, (train_idx, test_idx) in enumerate(kfold.split(ptids, trajectory_labels)):
            model = build_model(matchnet_config)
            optimizer = Adam(learning_rate=matchnet_config.learning_rate)
            model.compile(optimizer=optimizer)

            train_trajectories = ptids[train_idx]
            train_trajectory_labels = trajectory_labels[train_idx]
            test_trajectories = ptids[test_idx]
            
            train_trajectories, val_trajectories, _, _ = train_test_split(train_trajectories, train_trajectory_labels, test_size=0.25, stratify=train_trajectory_labels, random_state=42)
            
            # Prepare the data
            train_measurement_labels, train_true_labels, train_horizon_labels, train_metric_labels, train_windows, train_masks, train_lengths, train_patients = prepare_data(data_creator,
                study_df, missing_masks, train_trajectories, forwarded_indexes)
            val_measurement_labels, val_true_labels, _, val_metric_labels, val_windows, val_masks, val_lengths, val_patients = prepare_data(data_creator,
                study_df, missing_masks, val_trajectories, forwarded_indexes)
            test_measurement_labels, test_true_labels, _, test_metric_labels, test_windows, test_masks, test_lengths, test_patients = prepare_data(data_creator,
                study_df, missing_masks, test_trajectories, forwarded_indexes)

            if matchnet_config.oversampling:
                oversampler = RandomOverSampler(sampling_strategy=matchnet_config.oversample_ratio, random_state=42)
                
                # Create a filter for which samples to use for oversampling
                train_true_horizon_labels = list(map(lambda x: True if x == 1 or x == 0 else False, train_horizon_labels))
                train_true_horizon_labels = np.array(train_true_horizon_labels)
                train_idx = np.where(train_true_horizon_labels)[0].reshape(-1, 1)
                forwarded_idx = np.where(train_true_horizon_labels == False)[0]

                # Oversample the training data using the chosen oversample ratio
                train_idx = np.where(train_true_horizon_labels)[0].reshape(-1, 1)
                train_idx, _ = oversampler.fit_resample(train_idx, np.array(train_horizon_labels)[train_idx])
                train_idx = train_idx.ravel()

                # Readd forwarded indexes (no need for shuffling, as this is already done by the model fit by default)
                train_idx = np.concatenate([train_idx, forwarded_idx])

                # Use idx to oversample training data
                train_measurement_labels = train_measurement_labels[train_idx]
                train_true_labels = train_true_labels[train_idx]
                train_metric_labels = train_metric_labels[train_idx]
                train_windows = train_windows[train_idx]
                train_masks = train_masks[train_idx]
                train_lengths = train_lengths[train_idx]

            train_data = [train_windows, train_masks]
            train_labels = train_measurement_labels
            validation_data = [val_windows, val_masks]
            validation_labels = val_measurement_labels 

            # Compute proportional weights for train, val and test data
            train_lengths = median_traj_length / train_lengths
            val_lengths = median_traj_length / val_lengths
            test_lengths = median_traj_length / test_lengths

            model.fit(x=train_data, y=train_labels, batch_size=batch_size, epochs=max_epochs, sample_weight=[train_true_labels, train_metric_labels, train_lengths], validation_data=(
                validation_data, validation_labels, [val_true_labels, val_metric_labels, val_lengths]), validation_batch_size=len(val_true_labels), callbacks=[early_stopping])

            evaluation_results = model.evaluate([test_windows, test_masks], test_measurement_labels, sample_weight=[test_true_labels, test_metric_labels, test_lengths], batch_size=len(test_true_labels))
            evaluation_predictions = model.predict_on_batch([test_windows, test_masks])
            fold_c_index[cross_run] = compute_c_index_score(test_measurement_labels, evaluation_predictions, test_patients, test_metric_labels, pred_horizon=matchnet_config.pred_horizon)
            fold_au_rocs[cross_run] = evaluation_results[3]
            fold_au_prcs[cross_run] = evaluation_results[2]
            fold_conv[cross_run] = evaluation_results[1]

            # Compute calibration curves
            true_survival, pred_survival, all_true_probs, all_pred_probs = compute_calibration_curve(test_measurement_labels, evaluation_predictions, test_patients, test_metric_labels, pred_horizon=matchnet_config.pred_horizon, eval_time=eval_time)
            true_curves.append(true_survival)
            pred_curves.append(pred_survival)

            # Compute Brier score
            fold_brier_score[cross_run] = compute_brier_score(test_measurement_labels, evaluation_predictions, test_patients, test_metric_labels, pred_horizon=matchnet_config.pred_horizon, eval_time=eval_time)

        test_auroc[i] = np.mean(fold_au_rocs)
        test_auprc[i] = np.mean(fold_au_prcs)
        test_auroc_std[i] = np.std(fold_au_rocs)
        test_auprc_std[i] = np.std(fold_au_prcs)
        test_conv[i] = np.mean(fold_conv)
        test_c_idx[i] = np.mean(fold_c_index)
        test_brier_score[i] = np.mean(fold_brier_score)
        test_brier_std[i] = np.std(fold_brier_score)

        true_curves = np.array(true_curves)
        pred_curves = np.array(pred_curves)

        test_true_curves.append(np.mean(true_curves, axis=0))
        test_pred_curves.append(np.mean(pred_curves, axis=0))

    # Get best option
    best_index = np.argmax(test_conv)
    auroc = test_auroc[best_index]
    auprc = test_auprc[best_index]
    l1_l2_combination = combs[best_index]
    best_true_curve = test_true_curves[best_index]
    best_pred_curve = test_pred_curves[best_index]
    c_index = test_c_idx[best_index]

    print('========Evaluation results========')
    print(f'Best auroc: {auroc}, auprc: {auprc}, c-index: {c_index}, combination: {l1_l2_combination}')
    print(f'Test data auroc mean: {np.mean(test_auroc)}, std: {np.std(test_auroc)}')
    print(f'Test data auprc mean: {np.mean(test_auprc)}, std: {np.std(test_auprc)}')
    print(f'Test data c-index score mean: {np.mean(test_c_idx)}, std: {np.std(test_c_idx)}')
    print(f'Test data brier score mean: {np.mean(test_brier_score)}, std: {np.std(test_brier_score)}\n')

    # Save best calibration plot
    plt.plot(best_pred_curve, best_true_curve, marker='o', label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', c='black', label='Perfect calibration')
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed probability')
    plt.xlim(0.5, 1)
    plt.ylim(0.5, 1)
    plt.title(f"Calibration curve at t={eval_time * 6 + 6} months")
    plt.legend()

    filename = f'pred_h={matchnet_config.pred_horizon}, forwarding={matchnet_config.label_forwarding}, regularisation={matchnet_config.weight_regularisation}, oversampling={matchnet_config.oversampling}, eval_time={eval_time}'
    plt.savefig(f'{matchnet_config.output_path}/calibration{filename}.png')
    plt.close()

def prepare_data(data_creator: DataCreator, study_df: pd.DataFrame, missing_masks: pd.DataFrame, trajectories: List, forwarded_indexes: List):
    windows = study_df.loc[study_df['PTID'].isin(trajectories)]
    masks = missing_masks.loc[missing_masks['PTID'].isin(trajectories)]
    return data_creator.create_data(windows, masks, forwarded_indexes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Match-Net for a desired prediction horizon')
    parser.add_argument('--eval_time', type=int, help='Brier evaluation time')
    args = parser.parse_args()

    matchnet_config = MatchNetConfig(
        pred_horizon=1,
        window_length=5,
        cov_filters=512,
        mask_filters=8,
        cov_filter_size=6,
        mask_filter_size=6,
        cov_input_features=35,
        mask_input_features=35,
        dense_units=256,
        conv_blocks=1,
        dense_layers=1,
        dropout_rate=0.1,
        val_frequency=7,
        label_fowarding=True,
        weight_regularisation=False,
        oversampling=True,
        oversample_ratio=0.33,
        learning_rate=0.0001,
        output_path="output/oversampl_forwarding")

    train_model(matchnet_config, batch_size=32, eval_time=1)
