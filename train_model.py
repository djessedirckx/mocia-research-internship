from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np

from eda_preprocessing.DataPreprocessor import DataPreprocessor
from eda_preprocessing.TrainingDataCreator import TrainingDataCreator

def train(epochs: int):

    input_file = 'tadpole_challenge/TADPOLE_D1_D2.csv'
    data_preprocessor = DataPreprocessor(input_file, label_forwarding=False)
    data_creator = TrainingDataCreator(window_length=4, prediction_horizon=3)

    study_df, missing_masks = data_preprocessor.preprocess_data()

    # Make label binary, mark all Dementia instances as positive
    study_df['DX'] = study_df['DX'].replace('Dementia', 1)
    study_df['DX'] = study_df['DX'].replace(['MCI', 'NL', 'MCI to Dementia', 'NL to MCI', 'MCI to NL', 'Dementia to MCI', 'NL to Dementia'], 0)
    study_df['DX'].fillna(0, inplace=True)

    patients, horizon_labels, measurement_labels, feature_windows, mask_windows = data_creator.create_training_data(study_df, missing_masks)
    kfold_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kfold_split.split(feature_windows, horizon_labels):
        train_labels = horizon_labels[train_idx]
        test_labels = horizon_labels[test_idx]

        # Split train fold into train and validation data
        train_idx, val_idx, train_labels, val_labels = train_test_split(train_idx, train_labels, test_size=0.25, random_state=42, stratify=train_labels)
        train_windows = feature_windows[train_idx]
        val_windows = feature_windows[val_idx]
        test_windows = feature_windows[test_idx]


if __name__ == '__main__':
    epochs = 50
    train(epochs=epochs)
