from typing import Tuple, List


class MatchNetConfig():

    def __init__(self, pred_horizon: int,
                 window_length: int = None,
                 cov_filters: int = None,
                 mask_filters: int = None,
                 cov_filter_size: int = None,
                 mask_filter_size: int = None,
                 cov_input_features: int = None,
                 mask_input_features: int = None,
                 dense_units: int = None,
                 conv_blocks: int = None,
                 dense_layers: int = None,
                 learning_rate: float = None,
                 dropout_rate: float = None,
                 l1: float = None,
                 l2: float = None,
                 val_frequency: int = None,
                 mc_repeats: int = 10,
                 output_path: str = None,
                 label_fowarding: bool = False):

        # Model configuration
        self.cov_filters: int = cov_filters
        self.mask_filters: int = mask_filters
        self.cov_filter_size: int = cov_filter_size
        self.mask_filter_size: int = mask_filter_size
        self.dense_units: int = dense_units
        self.conv_blocks: int = conv_blocks
        self.dense_layers: int = dense_layers
        self.pred_horizon: int = pred_horizon
        self.window_length: int = window_length
        self.cov_input_shape: Tuple[int, int] = (self.window_length, cov_input_features)
        self.mask_input_shape: Tuple[int, int] = (self.window_length, mask_input_features)
        self.learning_rate: float = learning_rate
        self.dropout_rate: float = dropout_rate
        self.l1: float = l1
        self.l2: float = l2
        self.convergence_weights: List[Tuple[float, float]] = [
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1)
        ]
        self.val_frequency: int = val_frequency
        self.mc_repeats: int = mc_repeats
        self.output_path: str = output_path
        self.label_forwarding: bool = label_fowarding
