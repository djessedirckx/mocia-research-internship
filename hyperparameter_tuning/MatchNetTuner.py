from typing import List

import kerastuner as kt
import numpy as np
import pandas as pd
from tensorflow.keras import callbacks

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold

from eda_preprocessing.DataCreator import DataCreator


class MatchNetTuner(kt.Tuner):

    def __init__(self, oracle, hypermodel, prediction_horizon, max_model_size=None, optimizer=None, loss=None, metrics=None, distribution_strategy=None, directory=None, project_name=None, logger=None, tuner_id=None, overwrite=False):
        super(MatchNetTuner, self).__init__(oracle, hypermodel, max_model_size=max_model_size, optimizer=optimizer, loss=loss, metrics=metrics,
                                            distribution_strategy=distribution_strategy, directory=directory, project_name=project_name, logger=logger, tuner_id=tuner_id, overwrite=overwrite)

        self.prediction_horizon: int = prediction_horizon

    def run_trial(self, trial, trajectories, trajectory_labels, study_df, missing_masks):
        kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        hp = trial.hyperparameters

        # Define hyperparameter search ranges for batch size and early stopping patience
        batch_size = hp.Choice('batch_size', [32, 64, 128, 256, 512])
        stopping_patience = hp.Choice('stopping_patience', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        loss, au_roc, au_prc, convergence_metric = [], [], [], []
        for train_idx, val_idx in kfold.split(trajectories, trajectory_labels):
            train_trajectories, val_trajectories = trajectories[train_idx], trajectories[val_idx]

            # Initialise model and earlystopping callback
            model = self.hypermodel.build(trial.hyperparameters)
            early_stopping=EarlyStopping(monitor = 'val_convergence_metric', patience = stopping_patience, verbose = 1, mode = 'max')

            # Prepare the data
            window_length = model.layers[0].input_shape[0][1]
            data_creator = DataCreator(window_length, self.prediction_horizon)

            train_measurement_labels, train_true_labels, train_windows, train_masks = self.prepare_data(data_creator,
                study_df, missing_masks, train_trajectories)
            val_measurement_labels, val_true_labels, val_windows, val_masks = self.prepare_data(data_creator,
                study_df, missing_masks, val_trajectories)
            train_data = [train_windows, train_masks]
            train_labels = train_measurement_labels
            validation_data = [val_windows, val_masks]
            validation_labels = val_measurement_labels

            # TODO --> load epochs from config
            model.fit(x=train_data, y=train_labels, batch_size=batch_size, epochs=50, sample_weight=train_true_labels, validation_data=(
                validation_data, validation_labels, val_true_labels), validation_batch_size=len(val_true_labels), callbacks=[early_stopping])

            # Evaluate model
            evaluation_results = model.evaluate(
                validation_data, validation_labels, sample_weight=val_true_labels, batch_size=len(val_true_labels))
            loss.append(evaluation_results[0])
            au_roc.append(evaluation_results[3])
            au_prc.append(evaluation_results[2])
            convergence_metric.append(evaluation_results[1])

        # Store run
        self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(loss), 'val_au_roc': np.mean(
            au_roc), 'val_au_prc': np.mean(au_prc), 'val_convergence_metric': np.mean(convergence_metric)})
        self.save_model(trial.trial_id, model)

    def prepare_data(self, data_creator: DataCreator, study_df: pd.DataFrame, missing_masks: pd.DataFrame, trajectories: List):
        windows = study_df.loc[study_df['PTID'].isin(trajectories)]
        masks = missing_masks.loc[missing_masks['PTID'].isin(trajectories)]
        return data_creator.create_data(windows, masks)
