from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from eda_preprocessing.DataPreprocessor import DataPreprocessor
from eda_preprocessing.DataCreator import DataCreator
from model.MatchNetConfig import MatchNetConfig
from model.model_builder_short import build_model


def train(epochs: int, batch_size: int):

    input_file = 'tadpole_challenge/TADPOLE_D1_D2.csv'
    data_preprocessor = DataPreprocessor(input_file, label_forwarding=False)
    data_creator = DataCreator(window_length=4, prediction_horizon=3)

    study_df, missing_masks = data_preprocessor.preprocess_data()

    # Make label binary, mark all Dementia instances as positive
    study_df['DX'] = study_df['DX'].replace('Dementia', 1)
    study_df['DX'] = study_df['DX'].replace(
        ['MCI', 'NL', 'MCI to Dementia', 'NL to MCI', 'MCI to NL', 'Dementia to MCI', 'NL to Dementia'], 0)

    trajectory_labels, ptids = [], []
    for ptid, trajectory in study_df.groupby("PTID"):
        trajectory_labels.append(
            1) if 1 in trajectory['DX'].values else trajectory_labels.append(0)
        ptids.append(ptid)

    # Split data into train, val & test
    train_trajectories, test_trajectories, train_labels, test_labels = train_test_split(
        ptids, trajectory_labels, test_size=0.2, random_state=42, stratify=trajectory_labels)
    train_trajectories, val_trajectories = train_test_split(
        train_trajectories, test_size=0.25, random_state=42, stratify=train_labels)

    # Prepare training, validation & test data
    train_measurement_labels, train_true_labels, train_windows, train_masks = prepare_data(
        data_creator, study_df, missing_masks, train_trajectories)
    val_measurement_labels, val_true_labels, val_windows, val_masks = prepare_data(
        data_creator, study_df, missing_masks, val_trajectories)
    test_measurement_labels, test_true_labels, test_windows, test_masks = prepare_data(
        data_creator, study_df, missing_masks, test_trajectories)

    # Create MatchnetConfiguration
    convergence_weights = [
        (1, 1),
        (1, 1),
        (1, 1),
    ]

    model_config = MatchNetConfig(
        cov_filters=32,
        mask_filters=8,
        cov_filter_size=3,
        mask_filter_size=3,
        cov_input_shape=(4, 35),
        mask_input_shape=(4, 35),
        dense_units=32,
        pred_horizon=3,
        dropout_rate=0.2,
        convergence_weights=convergence_weights,
        val_frequency=5,
        val_score_repeats=10,
        output_path='output')

    optimizer = Adam()

    # Create the model
    model = build_model(model_config)
    model.compile(optimizer=optimizer)

    train_data = [train_windows, train_masks]
    train_labels = train_measurement_labels
    validation_data = [val_windows, val_masks]
    validation_labels = val_measurement_labels

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_convergence_metric', patience=model_config.val_frequency, verbose=1, mode='max')

    history = model.fit(x=train_data, y=train_labels, batch_size=batch_size, epochs=epochs, sample_weight=train_true_labels, validation_data=(
        validation_data, validation_labels, val_true_labels), validation_batch_size=len(val_true_labels), callbacks=[early_stopping])

    # Evaluate on test data
    print('Evaluating on test data...')
    model.evaluate([test_windows, test_masks], test_measurement_labels,
          sample_weight=test_true_labels, batch_size=len(test_true_labels))

    np.save('train_loss', history.history['loss'])
    np.save('val_loss', history.history['val_loss'])


def prepare_data(data_creator: DataCreator, study_df: pd.DataFrame, missing_masks: pd.DataFrame, trajectories: List):
    windows = study_df.loc[study_df['PTID'].isin(trajectories)]
    masks = missing_masks.loc[missing_masks['PTID'].isin(trajectories)]
    return data_creator.create_data(windows, masks)


def save_statistics(config: MatchNetConfig, model: Model, train_loss: List, val_loss: List, train_auroc: List, train_aurpc: List, val_auroc: List, val_aurpc: List):

    now = datetime.now().isoformat()

    # Save all statistics as binary numpy files
    np.save(f'{config.output_path}/train_loss-{now}', train_loss)
    np.save(f'{config.output_path}/val_loss-{now}', val_loss)
    np.save(f'{config.output_path}/train_auroc-{now}', train_auroc)
    np.save(f'{config.output_path}/train_aurpc-{now}', train_aurpc)
    np.save(f'{config.output_path}/val_auroc-{now}', val_auroc)
    np.save(f'{config.output_path}/val_aurpc-{now}', val_aurpc)

    # Store model
    model.save(f'{config.output_path}/model-{now}.hdf5')

    print(f'Stored output in {config.output_path}/, training finished')


if __name__ == '__main__':
    epochs = 50
    batch_size = 128
    train(epochs=epochs, batch_size=batch_size)
