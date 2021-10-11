import tensorflow as tf

from typing import Tuple

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model

class MatchNet(Model):
    
    def __init__(self, *args, **kwargs):
        super(MatchNet, self).__init__(*args, **kwargs)
        
        # Define loss and metric functions
        self.loss_fn = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.au_roc = AUC()
        self.au_prc = AUC(curve='PR')

        # Initialise trackers for metrics
        self.loss_tracker = Mean(name="loss")
        self.val_loss_tracker = Mean(name="val_loss")

        self.auroc_tracker = Mean(name="au_roc")
        self.val_auroc_tracker = Mean(name="val_au_roc")
        self.auprc_tracker = Mean(name="au_prc")
        self.val_auprc_tracker = Mean(name="val_auprc")
    
    def train_step(self, data):
        # Unpack the data
        measurements, labels, sample_weights = data
        
        with tf.GradientTape() as tape:
            # Execute forward pass
            predictions = self(measurements, training=True)
            
            # Compute loss
            loss = self.compute_loss(labels, predictions, sample_weights)
            
        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Compute metrics
        self.loss_tracker.update_state(loss)
        au_roc, au_prc = self.compute_metrics(labels, predictions, sample_weights)

        if au_roc != None:
            self.auroc_tracker.update_state(au_roc)
            self.auprc_tracker.update_state(au_prc)

        return {
            "loss": self.loss_tracker.result(),
            "au_roc": self.auroc_tracker.result(),
            "au_prc": self.auprc_tracker.result()
        } 

    def test_step(self, data):
        measurements, labels, sample_weights = data
        predictions = self(measurements, training=True)
        loss = self.compute_loss(labels, predictions, sample_weights)

        # Compute metrics
        self.val_loss_tracker.update_state(loss)
        au_roc, au_prc = self.compute_metrics(labels, predictions, sample_weights)

        if au_roc != None:
            self.val_auroc_tracker.update_state(au_roc)
            self.val_auprc_tracker.update_state(au_prc)

        return {
            "val_loss": self.val_loss_tracker.result(),
            "au_roc": self.val_auroc_tracker.result(),
            "au_prc": self.val_auprc_tracker.result()
        }    

    def compute_loss(self, labels, predictions, sample_weights):
        loss = []
        
        for i in range(len(predictions)):
            prediction = predictions[i]
            label = labels[:, i]
            weights = sample_weights[:, i]
            # imputed = imputed_labels[:, i]

            # Compute loss, ignore imputed (or forwarded) labels
            prediction = prediction[weights]
            label = label[weights]
            loss.append(self.loss_fn(label, prediction))

        loss = tf.concat(loss, 0)
        return tf.reduce_mean(loss)

    def compute_metrics(self, labels, predictions, sample_weights) -> Tuple[float, float]:
        au_roc, au_prc = 0, 0
        for i in range(len(predictions)):
            prediction = predictions[i]
            label = labels[:, i]
            weights = sample_weights[:, i]

            prediction = prediction[weights]
            label = label[weights]

            self.au_roc.update_state(label, prediction)
            au_roc += self.au_roc.result()
            self.au_prc.update_state(label, prediction)
            au_prc += self.au_prc.result()

            # Reset metrics
            self.au_roc.reset_states()
            self.au_prc.reset_states()

        return au_roc / len(predictions), au_prc / len(predictions)

    @property
    def metrics(self):
        return [self.loss_tracker, self.auroc_tracker, self.auprc_tracker]
