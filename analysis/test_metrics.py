from typing import List

import numpy as np
import pandas as pd

from lifelines import KaplanMeierFitter
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

def compute_survival_outcomes(events: np.array, predictions: np.array, ptids: List, true_labels: np.array, pred_horizon: int):
    true_times = []
    pred_times = []
    true_censoring = []
    pred_censoring = []

    if pred_horizon > 1:
        predictions = np.stack(predictions, axis=1)
    elif pred_horizon == 1:
        predictions = np.expand_dims(predictions, axis=1)

    for patient in np.unique(ptids):
        pt_idx = np.where(ptids == patient)[0]
        true_trajectory, pred_trajectory, pt_true_labels = restore_trajectory(pt_idx, events, predictions, true_labels, pred_horizon)

        # Get index (time) of right censoring or event occurence
        true_right_censored = 1.0 not in true_trajectory
        pred_right_censored = 1.0 not in pred_trajectory
        traj_length = np.where(pt_true_labels == True)[0][-1] if true_right_censored else np.where(true_trajectory == 1)[0][0]

        # Get index (time) of right censoring or predicted event occurence
        event_predicted = 1.0 in pred_trajectory
        if event_predicted:
            pred_index = np.where(pred_trajectory == 1)[0][0]
        else:
            pred_index = np.where(pt_true_labels == True)[0][-1]
    
        # Store data for c index computation
        true_times.append(traj_length)
        pred_times.append(pred_index)
        true_censoring.append(true_right_censored)
        pred_censoring.append(pred_right_censored)

    return true_times, pred_times, true_censoring, pred_censoring

def compute_c_index_score(events: np.array, predictions: np.array, ptids: List, true_labels: np.array, pred_horizon: int):
    true_times, pred_times, true_censoring, _ = compute_survival_outcomes(events, predictions, ptids, true_labels, pred_horizon)
    return concordance_index(true_times, pred_times, true_censoring)

def compute_survival_function(event_times: List, event_occured: np.array):
    # Fit the data to a Kaplan Meier curve for the different timesteps
    kmf = KaplanMeierFitter()
    kmf.fit(event_times, event_occured)
    return kmf.survival_function_

def add_survival_timesteps(surv_funct_1: pd.DataFrame, surv_func_2: pd.DataFrame) -> pd.DataFrame:
    last_1_time = surv_funct_1.index[-1]
    last_2_time = surv_func_2.index[-1]
    last_2_row = surv_func_2.iloc[-1]
    
    # Make sure the predicted and observed curve have an equal amount of steps
    for _ in range(int(last_1_time - last_2_time)):
        surv_func_2 = surv_func_2.append(last_2_row)

    return surv_func_2

def impute_missing_survival_steps(surv_func: pd.DataFrame):
    steps = np.arange(0, surv_func.index[-1] + 1)
    missing_steps = np.setdiff1d(steps, surv_func.index)
    
    # Impute missing steps by their previous value
    for step in missing_steps:
        previous_row = surv_func.xs(step - 1)
        previous_row.name = step
        surv_func = surv_func.append(previous_row)

    surv_func = surv_func.sort_index()
    return surv_func

def compute_calibration_curve(events: np.array, predictions: np.array, ptids: List, true_labels: np.array, pred_horizon: int):
    true_times, pred_times, true_censoring, pred_censoring = compute_survival_outcomes(events, predictions, ptids, true_labels, pred_horizon)

    # Compute true and predicted survival function
    true_survival_function = compute_survival_function(true_times, ~np.array(true_censoring))
    pred_survival_function = compute_survival_function(pred_times, ~np.array(pred_censoring))

    # Impute missing timesteps by using the previous valid timestep
    true_survival_function = impute_missing_survival_steps(true_survival_function)
    pred_survival_function = impute_missing_survival_steps(pred_survival_function)

    # Make sure both survival function end at the same timestep
    if len(true_survival_function) > len(pred_survival_function):
        pred_survival_function = add_survival_timesteps(true_survival_function, pred_survival_function)
    elif len(pred_survival_function) > len(true_survival_function):
        true_survival_function = add_survival_timesteps(pred_survival_function, true_survival_function)

    return true_survival_function['KM_estimate'].tolist() , pred_survival_function['KM_estimate'].tolist()