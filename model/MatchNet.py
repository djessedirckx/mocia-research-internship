import tensorflow as tf

from typing import Tuple

from sklearn.metrics import roc_auc_score, average_precision_score
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Mean
from tensorflow.keras.models import Model

from model.config.MatchNetConfig import MatchNetConfig


class MatchNet(Model):

    def __init__(self, config: MatchNetConfig, *args, **kwargs):
        super(MatchNet, self).__init__(*args, **kwargs)

        # Define loss and metric functions
        self.loss_fn = CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)

        # Initialise trackers for metrics
        self.loss_tracker = Mean(name="loss")
        self.val_loss_tracker = Mean(name="val_loss")

        self.val_auroc_tracker = Mean(name="val_au_roc")
        self.val_auprc_tracker = Mean(name="val_auprc")
        self.convergence_tracker = Mean(name="convergence_metric")

        # Store config
        self.config: MatchNetConfig = config

    def train_step(self, data):
        # Unpack the data
        measurements, labels, sample_weights = data
        train_weights = sample_weights[0]
        train_lengths = sample_weights[2]

        with tf.GradientTape() as tape:
            # Execute forward pass
            predictions = self(measurements, training=True)

            # Compute loss
            loss = self.compute_loss(labels, predictions, train_weights, train_lengths)

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute metrics
        self.loss_tracker.update_state(loss)

        return {
            "loss": self.loss_tracker.result()
        }

    def test_step(self, data):
        measurements, labels, sample_weights = data
        val_weights = sample_weights[0]
        metric_weights = sample_weights[1]
        val_lengths = sample_weights[2]
        loss_total, au_roc_total, au_prc_total, convergence_total = 0, 0, 0, 0

        # Compute test loss, auroc & auprc for a specified number of steps and
        # return average scores (mc dropout is applied)
        for i in range(self.config.mc_repeats):
            predictions = self(measurements, training=True)

            # Compute loss and metrics
            loss_total += self.compute_loss(labels,predictions, val_weights, val_lengths)
            au_roc, au_prc, convergence = self.compute_metrics(
                labels, predictions, metric_weights, self.config.convergence_weights)
            au_roc_total += au_roc
            au_prc_total += au_prc
            convergence_total += convergence

        # Update metrics
        self.val_loss_tracker.update_state(loss_total / self.config.mc_repeats)
        self.val_auroc_tracker.update_state(au_roc_total / self.config.mc_repeats)
        self.val_auprc_tracker.update_state(au_prc_total / self.config.mc_repeats)
        self.convergence_tracker.update_state(convergence_total / self.config.mc_repeats)

        return {
            "loss": self.val_loss_tracker.result(),
            "au_roc": self.val_auroc_tracker.result(),
            "au_prc": self.val_auprc_tracker.result(),
            "convergence_metric": self.convergence_tracker.result()
        }

    def compute_loss(self, labels, predictions, sample_weights, length_weights):
        loss = []

        # Iterate over each timepoint in the prediction horizon
        for i in range(labels.shape[1]):
            if self.config.pred_horizon == 1:
                prediction = predictions
            else:
                prediction = predictions[i]
            
            label = labels[:, i]
            weights = sample_weights[:, i]

            # Compute loss, ignore imputed (or forwarded-filled) labels
            prediction = prediction[weights]
            label = label[weights]
            length_weight = length_weights[weights]
            loss.append(self.loss_fn(label, prediction) * length_weight)

        loss = tf.concat(loss, 0)
        return tf.reduce_mean(loss)

    def compute_metrics(self, labels, predictions, sample_weights, convergence_weights) -> Tuple[float, float, float]:
        au_roc_total, au_prc_total, convergence = 0, 0, 0

        # Iterate over each timepoint in the prediction horizon
        for i in range(labels.shape[1]):
            if self.config.pred_horizon == 1:
                prediction = predictions[:, 1]
            else:
                prediction = predictions[i][:, 1]
            
            label = labels[:, i][:, 1]
            weights = sample_weights[:, i]
            beta, gamma = convergence_weights[i]

            prediction = prediction[weights]
            label = label[weights]

            # Compute auroc and auprc, ignore imputed (or forwarded-filled) labels
            au_roc = self.compute_au_roc(label, prediction)
            au_prc = self.compute_au_prc(label, prediction)

            au_roc_total += au_roc
            au_prc_total += au_prc

            convergence += (beta * au_roc + gamma * au_prc)

        # Return the average auroc and auprc for this prediction horizon
        return au_roc_total / labels.shape[1], au_prc_total / labels.shape[1], convergence / labels.shape[1]

    @property
    def metrics(self):
        return [self.loss_tracker, self.val_loss_tracker, self.val_auroc_tracker, self.val_auprc_tracker, self.convergence_tracker]

    @tf.function
    def compute_au_roc(self, labels, predictions):
        score = tf.numpy_function(roc_auc_score, [labels, predictions], tf.double)
        return score

    @tf.function
    def compute_au_prc(self, labels, predictions):
        score = tf.numpy_function(average_precision_score, [labels, predictions], tf.double)
        return score
