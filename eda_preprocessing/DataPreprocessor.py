import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple


class DataPreprocessor():

    def __init__(self, input_path: Path, label_forwarding: bool):
        self.input_path: Path = input_path
        self.label_forwarding = label_forwarding

    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
        study_df = self.load_data(self.input_path)
        study_df = study_df.sort_values(by=['Month'])

        # Remove entries without labels or starting AD meassurement
        invalid_entries = []
        for _, trajectory in study_df.groupby("PTID"):
            labels = trajectory['DX']

            # If trajectory has no labels, drop from dataset
            first_label_index = labels.first_valid_index()
            if first_label_index == None or labels.loc[first_label_index] == 'Dementia':
                invalid_entries.extend(trajectory.index.values)

        study_df.drop(invalid_entries, inplace=True)

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

        # Process positive labels after first stable diagnosis
        study_df, forwarded_indexes = self.process_positive_labels(study_df)

        print('Finished initial pre-processing of the data')

        return study_df, missing_masks, forwarded_indexes

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

        # Sanity checks
        assert (missing_masks == 1).equals(feature_set.isnull()), 'Incorrect missing values in masks'
        assert (missing_masks == 0).equals(feature_set.notnull()), 'Incorrect present values in masks'

        ohe_features = []
        for column in feature_set.columns[3:7]:
            one_hot = pd.get_dummies(feature_set[column], prefix=column, dummy_na=True)

            # Encode missing values as all ones
            one_hot.loc[one_hot[f'{column}_nan'] == 1, one_hot.columns] = np.ones(one_hot.shape[1])

            # Encode present values as all zeros
            one_hot.loc[one_hot[f'{column}_nan'] == 0, one_hot.columns] = np.zeros(one_hot.shape[1])     

            ohe_features.append(one_hot.iloc[:, :-1])

        ohe_features = pd.concat(ohe_features, axis=1)

        # Drop categorical columns and insert one_hot encoded masks
        missing_masks = missing_masks.drop(missing_masks.columns[3:7], axis=1)
        missing_masks = pd.concat([missing_masks.iloc[:, 0:3], ohe_features, missing_masks.iloc[:, 3:]], axis=1)

        return missing_masks

    def impute_data(self, study_df: pd.DataFrame, feature_set: pd.DataFrame) -> pd.DataFrame:
        print('Imputing missing values...')

        # Use zero-order interpolation on the data (execute per patient)
        for _, events in tqdm(study_df.groupby('PTID')):
            feature_set.loc[events.index, feature_set.columns] = events[feature_set.columns].fillna(
                method='ffill')

        # Fill remaining numerical column nan values with mean of all measurements
        for column in feature_set.columns[np.r_[0, 2:3, 7:22]]:
            feature_set[column].fillna(feature_set[column].mean(), inplace=True)

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
        ohe_features = pd.concat([pd.get_dummies(feature_set[column], prefix=column) for column in feature_set.columns[3:7]], axis=1)

        # Drop categorical columns and insert one_encoded features
        feature_set = feature_set.drop(feature_set.columns[3:7], axis=1)
        feature_set = pd.concat([feature_set.iloc[:, 0:3], ohe_features, feature_set.iloc[:, 3:]], axis=1)

        return feature_set

    def process_positive_labels(self, study_df: pd.DataFrame) -> pd.DataFrame:
        counter = 0
        all_forwarded_indexes = []

        for pt_id in study_df.loc[study_df['DX'] == 'Dementia']['PTID'].unique():

            # Get events for this patient
            events = study_df.loc[study_df['PTID'] == pt_id]

            # Get month of first stable diagnosis of AD
            ad_month = events.loc[events['DX'] == 'Dementia'].iloc[0]['Month']

            # Get indexes of measurements after first stable diagnosis
            forwarding_indexes = events.index[events['Month'] > ad_month]
            all_forwarded_indexes.extend(forwarding_indexes.tolist())

            if len(forwarding_indexes) > 0:

                counter += len(forwarding_indexes)

                # If specified, employ label forwarding
                if self.label_forwarding:
                    study_df.loc[forwarding_indexes, 'DX'] = study_df.loc[forwarding_indexes, 'DX'].fillna('Dementia')
                    continue
                
                # In case no label forwarding should be applied, remove timesteps after first stable diagnosis
                study_df = study_df.drop(index=forwarding_indexes)

        print(f'Processed {counter} positive timesteps')
        return study_df, all_forwarded_indexes
