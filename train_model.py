import itertools
import numpy as np
import tensorflow as tf

from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import Progbar
from sklearn.model_selection import StratifiedKFold, train_test_split

from eda_preprocessing.DataPreprocessor import DataPreprocessor
from eda_preprocessing.TrainingDataCreator import TrainingDataCreator
from model.model_builder import build_model

def train(folds, epochs: int, batch_size: int):

    input_file = 'tadpole_challenge/TADPOLE_D1_D2.csv'
    data_preprocessor = DataPreprocessor(input_file, label_forwarding=False)
    data_creator = TrainingDataCreator(window_length=4, prediction_horizon=3)

    study_df, missing_masks = data_preprocessor.preprocess_data()

    # Make label binary, mark all Dementia instances as positive
    study_df['DX'] = study_df['DX'].replace('Dementia', 1)
    study_df['DX'] = study_df['DX'].replace(['MCI', 'NL', 'MCI to Dementia', 'NL to MCI', 'MCI to NL', 'Dementia to MCI', 'NL to Dementia'], 0)
    
    # Impute missing labels and store index of imputed labels
    na_indexes = study_df.loc[study_df['DX'].isnull()].index
    study_df['DX'].fillna(0, inplace=True)

    # Get data for training and testing and specify the KFold split
    patients, horizon_labels, measurement_labels, feature_windows, mask_windows = data_creator.create_training_data(study_df, missing_masks)
    kfold_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print('Start training')
    for fold_step, (train_idx, test_idx) in enumerate(kfold_split.split(feature_windows, horizon_labels)):
        train_horizon_labels = horizon_labels[train_idx]

        # Split train fold into train and validation data
        train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42, stratify=train_horizon_labels)

        # Unpack training data from nested list
        train_measurement_labels = list(itertools.chain.from_iterable(measurement_labels[train_idx]))
        train_windows = np.array(list(itertools.chain.from_iterable(feature_windows[train_idx])))
        train_masks = np.array(list(itertools.chain.from_iterable(mask_windows[train_idx])))

        # val_measurement_labels = measurement_labels[val_idx]

        # Define array for retrieving feature windows and masks during training
        train_batch_idx = np.arange(0, len(train_windows))

        # Split training data into minibatches
        train_dataset = tf.data.Dataset.from_tensor_slices((train_batch_idx, train_measurement_labels))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        # Iterate for the specified number of epochs
        for epoch in range(epochs):
            print(f'epoch: {epoch+1}/{epochs} - fold: {fold_step+1}/{folds}\n')

            prog_bar = Progbar(target=len(train_batch_idx))

            # Iterate over the minibatches
            for step, (train_batch_data, train_batch_labels) in enumerate(train_dataset):
                batch_indexes = train_batch_data.numpy()

                windows = train_windows[batch_indexes]
                masks = train_masks[batch_indexes]

                # Update progress bar
                prog_bar.add(batch_size)


if __name__ == '__main__':
    folds = 5
    epochs = 50
    batch_size = 128
    train(folds=folds, epochs=epochs, batch_size=batch_size)
