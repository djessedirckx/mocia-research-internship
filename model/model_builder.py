from tensorflow.keras.layers import Conv1D, Dense, Dropout, Concatenate, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model

from model.MatchNetConfig import MatchNetConfig


def build_model(config: MatchNetConfig) -> Model:

    # Conv layers for covariates and masks
    covariate_input = Input(shape=config.cov_input_shape)
    covariate_conv = Conv1D(filters=config.cov_filters,
                            kernel_size=config.cov_filter_size, activation='relu')(covariate_input)

    mask_input = Input(shape=config.mask_input_shape)
    mask_conv = Conv1D(filters=config.mask_filters,
                       kernel_size=config.mask_filter_size, activation='relu')(mask_input)

    # Concatenate output from mask branch to main branch
    concat = Concatenate()([covariate_conv, mask_conv])

    # Second convolutional layer
    covariate_conv = Conv1D(filters=config.cov_filters,
                            kernel_size=config.cov_filter_size, activation='relu')(concat)
    mask_conv = Conv1D(filters=config.mask_filters,
                       kernel_size=config.mask_filter_size, activation='relu')(mask_conv)

    # Concatenate output from mask branch to main branch
    concat = Concatenate()([covariate_conv, mask_conv])

    # Dense layers
    dense = Dense(units=config.dense_units, activation='relu')(concat)
    dense = Dense(units=config.dense_units, activation='relu')(dense)

    # Define output layer and construct model
    output = Dense(units=config.pred_horizon, activation='softmax')

    model = Model(inputs=[covariate_input, mask_input], output=output)
    model.compile(optimizer=config.optimizer, loss=CategoricalCrossentropy())

    return model
