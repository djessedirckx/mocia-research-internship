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
from model.MatchNetConfig import MatchNetConfig
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
    # na_indexes = study_df.loc[study_df['DX'].isnull()].index
    # study_df['DX'].fillna(0, inplace=True)

    # Get data for training and testing and specify the KFold split
    patients, horizon_labels, measurement_labels, imputed_labels, feature_windows, mask_windows = data_creator.create_training_data(study_df, missing_masks)
    kfold_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Create MatchnetConfiguration
    model_config = MatchNetConfig(
        cov_filters=32, 
        mask_filters=8, 
        cov_filter_size=3, 
        mask_filter_size=3, 
        cov_input_shape=(4, 35), 
        mask_input_shape=(4, 35), 
        dense_units=32, 
        pred_horizon=3)

    optimizer = Adam()
    loss_fn = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

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

        # Create the model
        model = build_model(model_config)

        # Iterate for the specified number of epochs
        for epoch in range(epochs):
            print(f'epoch: {epoch+1}/{epochs} - fold: {fold_step+1}/{folds}\n')

            prog_bar = Progbar(target=len(train_batch_idx), stateful_metrics=['train_loss'])

            # Iterate over the minibatches
            for step, (train_batch_data, train_batch_labels) in enumerate(train_dataset):
                batch_indexes = train_batch_data.numpy()

                windows = train_windows[batch_indexes].astype('float32')
                masks = train_masks[batch_indexes].astype('float32')

                with tf.GradientTape() as tape:
                    predictions = model([windows, masks], training=True)

                    loss = []
                    for i in range(len(predictions)):
                        prediction = predictions[i]
                        label = train_batch_labels[:, i]

                        loss.append(loss_fn(label, prediction))

                    loss = tf.concat(loss, 0)
                    loss = tf.reduce_mean(loss)

                    # for prediction, label in zip(predictions, train_batch_labels):
                    #     loss = loss_fn(label, prediction)

                    # print(len(predictions))
                    # loss = loss_fn(train_batch_labels, predictions)
                    # loss_idx = tf.math.is_nan(loss)
                    # print(train_batch_labels)
                    # print(predictions)
                    # print(loss)
                    # loss = loss[~loss_idx]
                    # # loss = tf.reduce_mean(loss)

                gradients = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(gradients, model.trainable_weights))

                # Update progress bar
                prog_metrics = [('train_loss', loss)]
                prog_bar.add(batch_size, values=prog_metrics)


if __name__ == '__main__':
    folds = 5
    epochs = 50
    batch_size = 128
    train(folds=folds, epochs=epochs, batch_size=batch_size)
