from typing import Tuple


class RandomSearchConfig():

    def __init__(self, cov_input_shape: Tuple, mask_input_shape: Tuple, date: str) -> None:
        self.cov_input_shape = cov_input_shape
        self.mask_input_shape = mask_input_shape

        self.connected_layers = [1, 2, 3, 4, 5]
        self.conv_layers = [1, 2, 3, 4, 5]
        self.dropout_rate = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.lr = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        self.l1 = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        self.l2 = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        self.minibatch_size = [32, 64, 128, 256, 512]
        self.covariate_filters = [32, 64, 128, 256, 512]
        self.mask_filters = [8, 16, 32, 64, 128]
        self.oversample_ratio = [None, 1, 2, 3, 5, 10]
        self.connected_width = [32, 64, 128, 256, 512]
        self.conv_filter_width = [3, 4, 5, 6, 7, 8, 9, 10]

        # Add current datetime in isoformat to distinguish between runs
        self.output_folder = f'window_size_{cov_input_shape[0]}-{date}'
