import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Tuple


class TrainingDataCreator():

    def __init__(self, window_length: int, prediction_horizon: int) -> None:
        self.window_length = window_length
        self.prediction_horizon = prediction_horizon

    def create_training_data(self, study_data: pd.DataFrame, missing_masks: pd.DataFrame) -> Tuple[np.array, np.array, np.array]:
        labels, feature_window_set, mask_window_set = [], [], []

        print('Creating training data...')

        # Iterate over all patients in data
        for name, trajectory in tqdm(study_data.groupby("PTID")):
            traj_labels = trajectory['DX'].values
            mod = traj_labels.shape[0] % self.prediction_horizon
            max_divisble = traj_labels.shape[0] - mod
            traj_length = len(trajectory)

            # Fetch labels for desired predictions horizons
            pred_horizons = traj_labels[0:max_divisble].reshape(-1, self.prediction_horizon)

            # Fill right-censored labels with NaN if necessary
            if mod > 0:
                remainder = np.append(traj_labels[-mod:], np.full(self.prediction_horizon - mod, np.nan))
                remainder = np.expand_dims(remainder, axis=0)
                pred_horizons = np.concatenate((pred_horizons, remainder))

            # Extrapolate left-truncated values
            traj_features = self.extrapolate_values(trajectory)

            # Get masks and extrapolate
            masks = missing_masks.loc[missing_masks['PTID'] == name]
            masks_features = self.extrapolate_values(masks)

            traj_windows = []
            mask_windows = []
            for i in range(0, traj_length, self.prediction_horizon):
                traj_windows.append(traj_features[i:i+self.prediction_horizon+1, :])
                mask_windows.append(masks_features[i:i+self.prediction_horizon+1, :])

            labels.extend(pred_horizons)
            feature_window_set.extend(traj_windows)
            mask_window_set.extend(mask_windows)

        return np.array(labels), np.array(feature_window_set), np.array(mask_window_set)

    def extrapolate_values(self, trajectory: pd.DataFrame) -> np.array:
        traj_features = trajectory.iloc[:, 2:-1].values
        first_column = traj_features[0, :]
        extrapolated_values = np.tile(first_column, (self.window_length, 1))
        return np.concatenate((extrapolated_values, traj_features))
