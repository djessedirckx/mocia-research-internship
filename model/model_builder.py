from tensorflow.keras.layers import Conv1D, Dense, Dropout, Concatenate, Input, Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model

from model.MatchNetConfig import MatchNetConfig


def build_model(config: MatchNetConfig) -> Model:

    # Conv layers for covariates and masks
    covariate_input = Input(shape=config.cov_input_shape)
    covariate_conv = Conv1D(filters=config.cov_filters,
                            kernel_size=config.cov_filter_size, activation='relu', padding='causal')(covariate_input)

    mask_input = Input(shape=config.mask_input_shape)
    mask_conv = Conv1D(filters=config.mask_filters,
                       kernel_size=config.mask_filter_size, activation='relu', padding='causal')(mask_input)

    # Concatenate output from mask branch to main branch
    concat = Concatenate()([covariate_conv, mask_conv])

    # Second convolutional layer
    covariate_conv = Conv1D(filters=config.cov_filters,
                            kernel_size=config.cov_filter_size, activation='relu', padding='causal')(concat)
    mask_conv = Conv1D(filters=config.mask_filters,
                       kernel_size=config.mask_filter_size, activation='relu', padding='causal')(mask_conv)

    # Concatenate output from mask branch to main branch
    concat = Concatenate()([covariate_conv, mask_conv])

    # Dense layers
    flatten = Flatten()(concat)
    dense = Dense(units=config.dense_units, activation='relu')(flatten)
    dense = Dense(units=config.dense_units, activation='relu')(dense)

    # Define output layers based on specified prediction horizon
    output_layers = []
    for i in range(config.pred_horizon):
        output = Dense(units=2, name=f'output_{i}', activation='softmax')(dense)
        output_layers.append(output)

    # Construct and return model
    model = Model(inputs=[covariate_input, mask_input], outputs=output_layers)

    return model
