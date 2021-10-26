import itertools
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Tuple

from sklearn.preprocessing import OneHotEncoder


class DataCreator():

    def __init__(self, window_length: int, prediction_horizon: int) -> None:
        self.window_length = window_length
        self.prediction_horizon = prediction_horizon

    def create_data(self, study_data: pd.DataFrame, missing_masks: pd.DataFrame) -> Tuple[np.array, np.array, np.array, np.array]:
        measurement_labels, true_labels, feature_window_set, mask_window_set = [], [], [], []

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

            # Impute nan values. Store index of imputed labels for later loss calculation
            true_labels.append(~np.isnan(pred_horizons))
            pred_horizons = np.nan_to_num(pred_horizons)

            # Extrapolate left-truncated values
            traj_features = self.extrapolate_values(trajectory)

            # Get masks and extrapolate
            masks = missing_masks.loc[missing_masks['PTID'] == name]
            masks_features = self.extrapolate_values(masks)

            # Construct feature windows based on the desired prediction horizon
            traj_windows, mask_windows = [], []
            for i in range(0, traj_length, self.prediction_horizon):
                traj_windows.append(traj_features[i:i+self.window_length, :])
                mask_windows.append(masks_features[i:i+self.window_length, :])

            measurement_labels.extend(pred_horizons)
            feature_window_set.append(traj_windows)
            mask_window_set.append(mask_windows)

        # Apply one-hot encoding on measurement labels
        enc = OneHotEncoder()
        one_hot_labels = []
        for column in np.transpose(measurement_labels):
            one_hot = enc.fit_transform(column.reshape(-1, 1)).toarray()
            one_hot_labels.append(one_hot)

        one_hot_labels = np.concatenate(one_hot_labels, axis=1).reshape(len(measurement_labels), self.prediction_horizon, 2)
        true_labels = np.array(list(itertools.chain.from_iterable(true_labels)))
        feature_window_set = np.array(list(itertools.chain.from_iterable(feature_window_set)))
        mask_window_set = np.array(list(itertools.chain.from_iterable(mask_window_set)))

        return one_hot_labels, true_labels, feature_window_set, mask_window_set

    def extrapolate_values(self, trajectory: pd.DataFrame) -> np.array:
        traj_features = trajectory.iloc[:, 3:-1].values
        first_column = traj_features[0, :]
        extrapolated_values = np.tile(first_column, (self.window_length, 1))
        return np.concatenate((extrapolated_values, traj_features))
