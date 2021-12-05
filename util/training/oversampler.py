import numpy as np

from imblearn.over_sampling import RandomOverSampler

def oversample_train_data(oversample_ratio: float, train_horizon_labels: np.array, train_measurement_labels: np.array, train_true_labels: np.array,
                            train_metric_labels: np.array, train_windows: np.array, train_masks: np.array, train_lengths: np.array):
    oversampler = RandomOverSampler(sampling_strategy=oversample_ratio, random_state=42)
    
    # Create a filter for which samples to use for oversampling
    train_true_horizon_labels = list(map(lambda x: True if x == 1 or x == 0 else False, train_horizon_labels))
    train_true_horizon_labels = np.array(train_true_horizon_labels)
    train_idx = np.where(train_true_horizon_labels)[0].reshape(-1, 1)
    forwarded_idx = np.where(train_true_horizon_labels == False)[0]

    # Oversample the training data using the chosen oversample ratio
    train_idx = np.where(train_true_horizon_labels)[0].reshape(-1, 1)
    train_idx, _ = oversampler.fit_resample(train_idx, np.array(train_horizon_labels)[train_idx])
    train_idx = train_idx.ravel()

    # Readd forwarded indexes (no need for shuffling, as this is already done by the model fit by default)
    train_idx = np.concatenate([train_idx, forwarded_idx])

    # Use idx to oversample training data
    train_measurement_labels = train_measurement_labels[train_idx]
    train_true_labels = train_true_labels[train_idx]
    train_metric_labels = train_metric_labels[train_idx]
    train_windows = train_windows[train_idx]
    train_masks = train_masks[train_idx]
    train_lengths = train_lengths[train_idx]

    return train_measurement_labels, train_true_labels, train_metric_labels, train_windows, train_masks, train_lengths