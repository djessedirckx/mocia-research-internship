import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from typing import Tuple


class DataPreprocessor():

    def __init__(self, input_path: Path, label_forwarding: bool):
        self.input_path: Path = input_path
        self.label_forwarding = label_forwarding

    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        study_df = self.load_data(self.input_path)

        # Split data into meta and feature data
        meta_data = study_df.iloc[:, 0:2]
        months = study_df.iloc[:, -1]
        feature_set = study_df.iloc[:, 2:-1]

        # Make all missing values (-1 & -4 to NaN) equal
        # -1 == confirmed missing at point of data entry
        # -4 == passively’ missing or not applicable
        for column in feature_set.columns:
            feature_set[column] = feature_set[column].replace([-4, -1], np.nan)

        # Create missing masks
        missing_masks = self.construct_masking(feature_set)
        missing_masks = pd.concat([meta_data, missing_masks, months], axis=1)

        # Data imputation
        feature_set = self.impute_data(study_df, feature_set)

        # Numerical normalisation and encoding of categorical features
        feature_set = self.normalise_and_encode_data(feature_set)

        # Concat study dataframe
        study_df = pd.concat([meta_data, feature_set, months], axis=1)

        # If label forwarding is not applied, remove timesteps after first stable diagnosis of AD
        if not self.label_forwarding:
            study_df = self.remove_timesteps(study_df)

        # Reset index of both dataframes
        study_df.reset_index(inplace=True)
        missing_masks.reset_index(inplace=True)

        print('Finished initial pre-processing of the data')
        return study_df, missing_masks

    def load_data(self, input_path: Path) -> pd.DataFrame:
        print('Loading the data...')
        original_df = pd.read_csv(input_path)

        # Remove left-truncated patients
        non_truncated_events = original_df.loc[original_df['DX_bl'] != 'AD']

        # Select desired columns
        desired_columns = ['PTID', 'DX', 'AGE', 'APOE4', 'PTEDUCAT', 'PTETHCAT', 'PTGENDER', 'PTMARRY',
                           'PTRACCAT', 'Entorhinal', 'Fusiform', 'Hippocampus', 'ICV', 'MidTemp', 'Ventricles',
                           'WholeBrain', 'ADAS11', 'ADAS13', 'CDRSB', 'MMSE', 'RAVLT_forgetting',
                           'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_perc_forgetting', 'Month']

        study_df = non_truncated_events[desired_columns]
        study_df.columns = desired_columns

        return study_df

    def construct_masking(self, feature_set: pd.DataFrame) -> pd.DataFrame:
        print("Creating missingness masks...")
        missing_masks = feature_set.copy()

        # Replace all ones with zeros (1 means missing in the masked feature set)
        missing_masks = missing_masks.replace(1, 0)

        # Set all missing values to 1, present values to 0
        missing_masks = missing_masks.isnull().astype('int')

        assert (missing_masks == 1).equals(
            feature_set.isna()), 'Ḿasking is incorrect'

        # Apply one-hot encoding on categorical features
        # One-hot encode categorical features
        ohe_features = None

        for column in feature_set.columns[3:7]:
            encoded_feature = pd.get_dummies(
                feature_set[column], prefix=column, dummy_na=True)

            # Encode missing values as all ones
            encoded_feature.loc[encoded_feature[f'{column}_nan'] == 1] = np.ones(
                encoded_feature.shape[1])

            # Encode present values as all zeros
            encoded_feature.loc[encoded_feature[f'{column}_nan'] == 1] = np.zeros(
                encoded_feature.shape[1])

            ohe_features = pd.concat(
                [ohe_features, encoded_feature.iloc[:, :-1]], axis=1)

        # Concat one-hot encoded features and drop original categorical columns
        missing_masks = pd.concat([missing_masks, ohe_features], axis=1)
        missing_masks = missing_masks.drop(missing_masks.columns[3:7], axis=1)

        return missing_masks

    def impute_data(self, study_df: pd.DataFrame, feature_set: pd.DataFrame) -> pd.DataFrame:
        print('Imputing missing values...')

        # Use zero-order interpolation on the data (execute per patient)
        for pt in tqdm(study_df['PTID'].unique()):
            events = study_df.loc[study_df['PTID'] == pt]
            feature_set.loc[events.index, feature_set.columns] = events[feature_set.columns].fillna(
                method='ffill')

        # Fill remaining numerical column nan values with mean of all measurements
        for column in feature_set.columns[np.r_[0, 2:3, 7:22]]:
            feature_set[column].fillna(
                feature_set[column].mean(), inplace=True)

        # Apply data imputation on APOE4 column. Values are replaced based on their probability of occurence
        apoe4_stats = feature_set['APOE4'].value_counts()
        apoe4_stats = apoe4_stats / len(feature_set)

        rng = np.random.default_rng(seed=42)
        nan_apoe_rows = feature_set.index[feature_set['APOE4'].isna()]

        for nan_apoe in nan_apoe_rows:
            rnd = rng.random()

            if rnd < apoe4_stats[0]:
                feature_set.loc[nan_apoe, 'APOE4'] = 0
            elif rnd >= apoe4_stats[0] and rnd <= apoe4_stats[1]:
                feature_set.loc[nan_apoe, 'APOE4'] = 1
            else:
                feature_set.loc[nan_apoe, 'APOE4'] = 2

        return feature_set

    def normalise_and_encode_data(self, feature_set: pd.DataFrame) -> pd.DataFrame:
        print('Normalising and encoding data...')
        # Normalize numerical features
        for column in feature_set.columns[np.r_[0:3, 7:22]]:
            feature_set[column] = (
                feature_set[column] - feature_set[column].mean()) / feature_set[column].std()

        # One-hot encode categorical features
        for column in feature_set.columns[3:7]:
            feature_set = pd.concat([feature_set, pd.get_dummies(
                feature_set[column], prefix=column)], axis=1)

        # Drop categorical columns
        feature_set = feature_set.drop(feature_set.columns[3:7], axis=1)

        return feature_set

    def remove_timesteps(self, study_df: pd.DataFrame) -> pd.DataFrame:
        counter = 0

        # Iterate over all patients with a stable AD diagnosis and remove redundant measurements
        for pt_id in study_df[study_df['DX'] == 'Dementia']['PTID'].unique():

            # Get events for this patient
            events = study_df.loc[study_df['PTID'] == pt_id]

            # Get index of first stable diagnosis of AD
            ad_index = events.index[events['DX'] == 'Dementia'][0]

            # Get indexes of measurements after first stable diagnosis
            forwarding_indexes = events.index[events.index > ad_index]

            # Remove all measurements after first stable diagnosis
            if len(forwarding_indexes) > 0:
                study_df = study_df.drop(index=forwarding_indexes)
                counter += len(forwarding_indexes)

        print(f'Removed {counter} redundant timesteps')
        return study_df
