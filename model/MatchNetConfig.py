from typing import Tuple, List


class MatchNetConfig():

    def __init__(self, cov_filters: int,
                 mask_filters: int,
                 cov_filter_size: int,
                 mask_filter_size: int,
                 cov_input_shape: Tuple[int, int],
                 mask_input_shape: Tuple[int, int],
                 dense_units: int,
                 pred_horizon: int,
                 dropout_rate: float,
                 l1: float,
                 l2: float,
                 convergence_weights: List[Tuple[float, float]],
                 val_frequency: int,
                 val_score_repeats: int,
                 output_path: str):

        # Model configuration
        self.cov_filters: int = cov_filters
        self.mask_filters: int = mask_filters
        self.cov_filter_size: int = cov_filter_size
        self.mask_filter_size: int = mask_filter_size
        self.cov_input_shape: Tuple[int, int] = cov_input_shape
        self.mask_input_shape: Tuple[int, int] = mask_input_shape
        self.dense_units: int = dense_units
        self.pred_horizon: int = pred_horizon
        self.dropout_rate: float = dropout_rate
        self.l1: float = l1
        self.l2: float = l2
        self.convergence_weights: List[Tuple[float, float]] = convergence_weights
        self.val_frequency = val_frequency
        self.val_score_repeats: int = val_score_repeats
        self.output_path: str = output_path
