from typing import List

import numpy as np
import pandas as pd

from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sksurv.metrics import brier_score

def restore_trajectory(pt_idx: List, events: np.array, predictions: np.array, true_labels: np.array, pred_horizon: int, return_probs: bool=False):
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
            true_trajectory[counter] = true_event[i][1] if return_probs else np.argmax(true_event[i])
            pred_trajectory[counter] = prediction[i][1] if return_probs else np.argmax(prediction[i])
            true_unpacked_labels[counter] = true_label[i]
            counter += 1

    return true_trajectory, pred_trajectory, true_unpacked_labels

def compute_survival_outcomes(events: np.array, predictions: np.array, ptids: List, true_labels: np.array, pred_horizon: int, all_ids):
    true_times = []
    pred_times = []
    true_censoring = []
    pred_censoring = []

    for patient in np.unique(ptids):
        pt_idx = np.where(all_ids == patient)[0]
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
    if pred_horizon > 1:
        predictions = np.stack(predictions, axis=1)
    elif pred_horizon == 1:
        predictions = np.expand_dims(predictions, axis=1)

    true_times, pred_times, true_censoring, _ = compute_survival_outcomes(events, predictions, ptids, true_labels, pred_horizon, ptids)
    return concordance_index(true_times, pred_times, true_censoring)

def compute_survival_function(event_times: List, event_occured: np.array):
    # Fit the data to a Kaplan Meier curve for the different timesteps
    kmf = KaplanMeierFitter()
    kmf.fit(event_times, event_occured)
    return kmf.survival_function_

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

def compute_calibration_curve(events: np.array, predictions: np.array, ptids: List, true_labels: np.array, pred_horizon: int, eval_time: int):  
    
    # Restore timeline (in terms of prediction and observations)
    if pred_horizon > 1:
        predictions = np.stack(predictions, axis=1)
    elif pred_horizon == 1:
        predictions = np.expand_dims(predictions, axis=1)

    prob_at_eval_time = []
    ptids_at_eval_time = []

    for patient in np.unique(ptids):
        pt_idx = np.where(ptids == patient)[0]
        true_probs, pred_probs, pt_true_labels = restore_trajectory(pt_idx, events, predictions, true_labels, pred_horizon, return_probs=True)

        # Only include patients that are still in the study at this point and whose predictions
        # at time of evaluation are not imputed
        if len(true_probs) >= eval_time + 1 and pt_true_labels[eval_time]:
            prob_at_eval_time.append(pred_probs[eval_time])
            ptids_at_eval_time.append(patient)

    true_times, _, true_censoring, _ = compute_survival_outcomes(events, predictions, ptids, true_labels, pred_horizon, ptids)

    sorted_index = np.argsort(prob_at_eval_time)
    prob_at_eval_time = np.array(prob_at_eval_time)[sorted_index]
    ptids_at_eval_time = np.array(ptids_at_eval_time)[sorted_index]

    # Split into 5 equally sized strata
    prob_at_eval_time = np.array_split(prob_at_eval_time, 5)
    ptids_at_eval_time = np.array_split(ptids_at_eval_time, 5)

    return_true_probs = np.zeros(5)
    return_pred_probs = np.zeros(5)

    for i in range(5):
        # Probability of not getting diagnosed
        pred_probs = 1 - prob_at_eval_time[i]
        patients = ptids_at_eval_time[i]
        
        # Compute survival function (kaplan meier curve) for observed values
        true_times, _, true_censoring, _ = compute_survival_outcomes(events, predictions, patients, true_labels, pred_horizon, ptids)
        true_survival_function = compute_survival_function(true_times, ~np.array(true_censoring))
        true_survival_function = impute_missing_survival_steps(true_survival_function)

        # Store result --> observed probability at evaluation time t and mean predicted probability
        # at observation time t
        return_true_probs[i] = true_survival_function.loc[eval_time]
        return_pred_probs[i] = np.mean(pred_probs)

    return return_true_probs, return_pred_probs
    
def compute_brier_score(events: np.array, predictions: np.array, ptids: List, true_labels: np.array, pred_horizon: int, eval_time: int) -> float:

    if pred_horizon > 1:
        predictions = np.stack(predictions, axis=1)
    elif pred_horizon == 1:
        predictions = np.expand_dims(predictions, axis=1)

    # Compute true survival times and detect whether trajectory is censored
    true_times, _, true_censoring, _ = compute_survival_outcomes(events, predictions, ptids, true_labels, pred_horizon, ptids)

    # Restore sets of predictions (prediction horizons) into one trajectory per patients
    pred_trajectories, pred_labels = [], []
    for patient in np.unique(ptids):
        pt_idx = np.where(ptids == patient)[0]
        _, pred_trajectory, pred_label = restore_trajectory(pt_idx, events, predictions, true_labels, pred_horizon, return_probs=True)
        pred_trajectories.append(pred_trajectory)
        pred_labels.append(pred_label)

    # Get all patients that are still part of the study at time of evaluation
    cal_true_times, cal_cens, cal_preds = [], [], []
    for time, censoring, prediction, label in zip(true_times, true_censoring, pred_trajectories, pred_labels):
        if time >= eval_time and label[eval_time]: 
            cal_true_times.append(time)

            # brier_score implementation requires boolean that event happened (not censored) and 
            # probability that event did not occur yet
            cal_cens.append(not censoring)
            cal_preds.append(1 - prediction)

    # Get predictions at eval_time and store true diagnosis/censoring time in desired format
    cal_predictions = [p[eval_time] for p in cal_preds]
    y = np.array(list(zip(cal_cens, cal_true_times)), dtype=[('cens', '?'), ('time', '<f8')])

    # Compute and return brier score weighted for censoring
    return brier_score(y, y, cal_predictions, eval_time)[1][0]