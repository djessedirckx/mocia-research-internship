from typing import Tuple, List


class MatchNetConfig():

    def __init__(self, pred_horizon: int,
                 cov_filters: int = None,
                 mask_filters: int = None,
                 cov_filter_size: int = None,
                 mask_filter_size: int = None,
                 cov_input_shape: Tuple[int, int] = None,
                 mask_input_shape: Tuple[int, int] = None,
                 dense_units: int = None,
                 conv_blocks: int = None,
                 dense_layers: int = None,
                 dropout_rate: float = None,
                 l1: float = None,
                 l2: float = None,
                 convergence_weights: List[Tuple[float, float]] = None,
                 val_frequency: int = None,
                 mc_repeats: int = 10,
                 output_path: str = None):

        # Model configuration
        self.cov_filters: int = cov_filters
        self.mask_filters: int = mask_filters
        self.cov_filter_size: int = cov_filter_size
        self.mask_filter_size: int = mask_filter_size
        self.cov_input_shape: Tuple[int, int] = cov_input_shape
        self.mask_input_shape: Tuple[int, int] = mask_input_shape
        self.dense_units: int = dense_units
        self.conv_blocks: int = conv_blocks
        self.dense_layers: int = dense_layers
        self.pred_horizon: int = pred_horizon
        self.dropout_rate: float = dropout_rate
        self.l1: float = l1
        self.l2: float = l2
        self.convergence_weights: List[Tuple[float,
                                             float]] = convergence_weights
        self.val_frequency: int = val_frequency
        self.mc_repeats: int = mc_repeats
        self.output_path: str = output_path
