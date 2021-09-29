from tensorflow.keras.optimizers import Adam

from typing import Tuple


class MatchNetConfig():

    def __init__(self, cov_filters: int,
                 mask_filters: int,
                 cov_filter_size: int,
                 mask_filter_size: int,
                 cov_input_shape: Tuple[int, int],
                 mask_input_shape: Tuple[int, int],
                 dense_units: int,
                 pred_horizon: int,
                 lr_rate: float,
                 batch_size: int):

        # Model configuration
        self.cov_filters: int = cov_filters
        self.mask_filters: int = mask_filters
        self.cov_filter_size: int = cov_filter_size
        self.mask_filter_size: int = mask_filter_size
        self.cov_input_shape: Tuple[int, int] = cov_input_shape
        self.mask_input_shape: Tuple[int, int] = mask_input_shape
        self.dense_units: int = dense_units
        self.pred_horizon: int = pred_horizon

        # Training configuration
        self.optimizer = Adam(learning_rate=lr_rate)
        self.batch_size = batch_size