from typing import List

import kerastuner as kt
import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from eda_preprocessing.DataCreator import DataCreator


class MatchNetTuner(kt.Tuner):

    def __init__(self, oracle, hypermodel, prediction_horizon, max_model_size=None, optimizer=None, loss=None, metrics=None, distribution_strategy=None, directory=None, project_name=None, logger=None, tuner_id=None, overwrite=False):
        super(MatchNetTuner, self).__init__(oracle, hypermodel, max_model_size=max_model_size, optimizer=optimizer, loss=loss, metrics=metrics,
                                            distribution_strategy=distribution_strategy, directory=directory, project_name=project_name, logger=logger, tuner_id=tuner_id, overwrite=overwrite)

        self.prediction_horizon: int = prediction_horizon

    def run_trial(self, trial, trajectories, trajectory_labels, study_df, missing_masks, forwarded_indexes, oversampling=False):
        hp = trial.hyperparameters

        # Define hyperparameter search ranges for batch size and early stopping patience
        batch_size = hp.Choice('batch_size', [32, 64, 128, 256, 512])
        stopping_patience = hp.Choice('stopping_patience', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Split data into training and validation
        train_trajectories, val_trajectories, _, _ = train_test_split(trajectories, trajectory_labels, test_size=0.25, stratify=trajectory_labels, random_state=42)

        # Initialise model and earlystopping callback
        model = self.hypermodel.build(trial.hyperparameters)
        early_stopping=EarlyStopping(monitor = 'val_convergence_metric', patience = stopping_patience, verbose = 1, mode = 'max')

        # Initialise the data creator for preparing the data using the chosen window length
        window_length = model.layers[0].input_shape[0][1]
        data_creator = DataCreator(window_length, self.prediction_horizon)

        # Prepare the data
        train_measurement_labels, train_true_labels, train_horizon_labels, train_metric_labels, train_windows, train_masks = self.prepare_data(data_creator,
            study_df, missing_masks, train_trajectories, forwarded_indexes)
        val_measurement_labels, val_true_labels, _, val_metric_labels, val_windows, val_masks = self.prepare_data(data_creator,
            study_df, missing_masks, val_trajectories, forwarded_indexes)

        if oversampling:
            oversample_ratio = hp.Choice('oversample_ratio', [1.0, 0.5, 0.33, 0.2, 0.1])
            oversampler = RandomOverSampler(sampling_strategy=oversample_ratio, random_state=42)
            
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

        train_data = [train_windows, train_masks]
        train_labels = train_measurement_labels
        validation_data = [val_windows, val_masks]
        validation_labels = val_measurement_labels

        # TODO --> load epochs from config
        model.fit(x=train_data, y=train_labels, batch_size=batch_size, epochs=50, sample_weight=[train_true_labels, train_metric_labels], validation_data=(
            validation_data, validation_labels, [val_true_labels, val_metric_labels]), validation_batch_size=len(val_true_labels), callbacks=[early_stopping])

        # Evaluate model on validation data
        evaluation_results = model.evaluate(validation_data, validation_labels, sample_weight=[val_true_labels, val_metric_labels], batch_size=len(val_true_labels))
        loss = evaluation_results[0]
        au_roc = evaluation_results[3]
        au_prc = evaluation_results[2]
        convergence_metric = evaluation_results[1]

        # Store run
        self.oracle.update_trial(trial.trial_id, {'val_loss': loss, 'val_au_roc': au_roc, 'val_au_prc': au_prc, 'val_convergence_metric': convergence_metric})
        self.save_model(trial.trial_id, model)

    def prepare_data(self, data_creator: DataCreator, study_df: pd.DataFrame, missing_masks: pd.DataFrame, trajectories: List, forwarded_indexes: List):
        windows = study_df.loc[study_df['PTID'].isin(trajectories)]
        masks = missing_masks.loc[missing_masks['PTID'].isin(trajectories)]
        return data_creator.create_data(windows, masks, forwarded_indexes)
