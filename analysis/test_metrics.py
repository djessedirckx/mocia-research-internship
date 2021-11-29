from typing import List

import numpy as np

from lifelines.utils import concordance_index

def restore_trajectory(pt_idx: List, events: np.array, predictions: np.array, true_labels: np.array, pred_horizon):
    pt_events = events[pt_idx]
    pt_predictions = predictions[pt_idx]
    pt_true_labels = true_labels[pt_idx]
    
    true_trajectory = np.empty(len(pt_idx) * pred_horizon)
    pred_trajectory = np.empty(len(pt_idx) * pred_horizon)
    true_unpacked_labels = np.empty(len(pt_idx) * pred_horizon)

    # Restore trajectory from cutted prediction horizons
    counter = 0
    for true_event, prediction, true_label in zip(pt_events, pt_predictions, pt_true_labels):
        for i in range(pred_horizon):
            true_trajectory[counter] = np.argmax(true_event[i])
            pred_trajectory[counter] = np.argmax(prediction[i])
            true_unpacked_labels[counter] = true_label[i]
            counter += 1

    return true_trajectory, pred_trajectory, true_unpacked_labels

def compute_c_index_score(events: np.array, predictions: np.array, ptids: List, true_labels: np.array, pred_horizon: int):
    true_times = []
    pred_times = []
    censoring = []

    if pred_horizon > 1:
        predictions = np.stack(predictions, axis=1)

    for patient in np.unique(ptids):
        pt_idx = np.where(ptids == patient)[0]
        true_trajectory, pred_trajectory, pt_true_labels = restore_trajectory(pt_idx, events, predictions, true_labels, pred_horizon)

        # Get index (time) of right censoring or event occurence
        right_censored = 1.0 not in true_trajectory
        traj_length = np.where(pt_true_labels == True)[0][-1] if right_censored else np.where(true_trajectory == 1)[0][0]

        # Get index (time) of right censoring or predicted event occurence
        event_predicted = 1.0 in pred_trajectory
        if event_predicted:
            pred_index = np.where(pred_trajectory == 1)[0][0]
        else:
            pred_index = np.where(pt_true_labels == True)[0][-1]

        # Store data for c index computation
        true_times.append(traj_length)
        pred_times.append(pred_index)
        censoring.append(right_censored)

    return concordance_index(true_times, pred_times, censoring)