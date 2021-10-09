from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import Progbar
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from eda_preprocessing.DataPreprocessor import DataPreprocessor
from eda_preprocessing.DataCreator import DataCreator
from model.MatchNetConfig import MatchNetConfig
from model.model_builder import build_model

def train(epochs: int, batch_size: int):

    input_file = 'tadpole_challenge/TADPOLE_D1_D2.csv'
    data_preprocessor = DataPreprocessor(input_file, label_forwarding=False)
    data_creator = DataCreator(window_length=4, prediction_horizon=3)

    study_df, missing_masks = data_preprocessor.preprocess_data()

    # Make label binary, mark all Dementia instances as positive
    study_df['DX'] = study_df['DX'].replace('Dementia', 1)
    study_df['DX'] = study_df['DX'].replace(['MCI', 'NL', 'MCI to Dementia', 'NL to MCI', 'MCI to NL', 'Dementia to MCI', 'NL to Dementia'], 0)
    
    trajectory_labels, ptids = [], []
    for ptid, trajectory in study_df.groupby("PTID"):
        trajectory_labels.append(1) if 1 in trajectory['DX'].values else trajectory_labels.append(0)
        ptids.append(ptid)

    # Split data into train, val & test
    train_trajectories, test_trajectories, train_labels, test_labels = train_test_split(ptids, trajectory_labels, test_size=0.2, random_state=42, stratify=trajectory_labels)
    train_trajectories, val_trajectories = train_test_split(train_trajectories, test_size=0.25, random_state=42, stratify=train_labels)
   
    # Prepare training, validation & test data
    train_measurement_labels, train_imputed_labels, train_windows, train_masks = prepare_data(data_creator, study_df, missing_masks, train_trajectories)
    val_measurement_labels, val_imputed_labels, val_windows, val_masks = prepare_data(data_creator, study_df, missing_masks, val_trajectories)
    test_measurement_labels, test_imputed_labels, test_windows, test_masks = prepare_data(data_creator, study_df, missing_masks, test_trajectories)

    # Create MatchnetConfiguration
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
        val_score_repeats=10)

    optimizer = Adam()
    loss_fn = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    # Define array for retrieving feature windows and masks during training
    train_batch_idx = np.arange(0, len(train_windows))
    val_idx = np.arange(0, len(val_windows))

    # Split training data into minibatches
    train_dataset = tf.data.Dataset.from_tensor_slices((train_batch_idx, train_measurement_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Create the model
    model = build_model(model_config)

    # Training stats
    best_val_loss = np.inf
    best_model_weights = None

    # Iterate for the specified number of epochs
    for epoch in range(epochs):
        print(f'epoch: {epoch+1}/{epochs}\n')

        prog_bar = Progbar(target=len(train_batch_idx), stateful_metrics=['train_loss', 'AUROC', 'AUPRC'])

        # Iterate over the minibatches
        for step, (train_batch_data, train_batch_labels) in enumerate(train_dataset):
            batch_indexes = train_batch_data.numpy()

            # Get boolean array that represents which labels are imputed
            batch_imputed_labels = train_imputed_labels[batch_indexes]

            windows = train_windows[batch_indexes].astype('float32')
            masks = train_masks[batch_indexes].astype('float32')

            with tf.GradientTape() as tape:
                predictions = model([windows, masks], training=True)
                loss = compute_loss(loss_fn, train_batch_labels, predictions, batch_imputed_labels)

            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            # Compute metrics
            au_roc, au_prc = compute_metrics(train_batch_labels, predictions)
            if au_roc != None and au_prc != None:
                prog_metrics = [('train_loss', loss), ('AUROC', au_roc), ('AUPRC', au_prc)]
            else:
                prog_metrics = [('train_loss', loss)]

            # Update progress bar
            prog_bar.add(batch_size, values=prog_metrics)

        if epoch % 10 == 0:
            # Compute validation performance

            val_loss, val_au_roc, val_au_prc = 0, 0, 0
            for i in range(model_config.val_score_repeats):
                val_predictions = model([val_windows, val_masks])

                val_loss += compute_loss(loss_fn, val_measurement_labels, val_predictions, val_imputed_labels)
                val_au_roc_sample, val_au_prc_sample = compute_metrics(val_measurement_labels, val_predictions)
                val_au_roc += val_au_roc_sample
                val_au_prc += val_au_prc_sample

            val_loss /= model_config.val_score_repeats
            val_au_roc /= model_config.val_score_repeats
            val_au_prc /= model_config.val_score_repeats

            print(f'Validation loss: {val_loss}, AUROC: {val_au_roc}, AUPRC: {val_au_prc}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = model.get_weights()

def prepare_data(data_creator: DataCreator, study_df: pd.DataFrame, missing_masks: pd.DataFrame, trajectories: List):
    windows = study_df.loc[study_df['PTID'].isin(trajectories)]
    masks = missing_masks.loc[missing_masks['PTID'].isin(trajectories)]
    return data_creator.create_data(windows, masks)

def compute_loss(loss_fn, labels, predictions, imputed_labels):
    loss = []
    
    for i in range(len(predictions)):
        prediction = predictions[i]
        label = labels[:, i]
        imputed = imputed_labels[:, i]

        # Compute loss, ignore imputed (or forwarded) labels
        prediction = prediction[imputed == False]
        label = label[imputed == False]
        loss.append(loss_fn(label, prediction))

    loss = tf.concat(loss, 0)
    return tf.reduce_mean(loss)

def compute_metrics(labels, predictions) -> Tuple[int, int]:
    au_roc, au_rpc, decay = 0, 0, 0
    for i in range(len(predictions)):
        prediction = predictions[i]
        label = labels[:, i]

        # Compute AUROC and AURPC score. Try except is added in case only one class is present
        # for the respective timepoint in the minibatch
        try:
            au_roc += roc_auc_score(label, prediction)
            au_rpc += average_precision_score(label, prediction)
        except:
            decay += 1

    if len(predictions) - decay != 0:
        return au_roc / (len(predictions) - decay), au_rpc / (len(predictions) - decay)
    return None, None

if __name__ == '__main__':
    epochs = 50
    batch_size = 128
    train(epochs=epochs, batch_size=batch_size)
