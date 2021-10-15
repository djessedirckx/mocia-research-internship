from tensorflow.keras.layers import Conv1D, Dense, Concatenate, Input, Flatten

from model.MatchNet import MatchNet
from model.MatchNetConfig import MatchNetConfig
from model.layers.MCDropout import MCDropout


def build_model(config: MatchNetConfig) -> MatchNet:

    # Conv layers for covariates and masks
    covariate_input = Input(shape=config.cov_input_shape)
    covariate_conv = Conv1D(filters=config.cov_filters,
                            kernel_size=config.cov_filter_size, activation='relu', padding='causal')(covariate_input)
    covariate_dropout = MCDropout(rate=config.dropout_rate)(covariate_conv)    

    mask_input = Input(shape=config.mask_input_shape)
    mask_conv = Conv1D(filters=config.mask_filters,
                       kernel_size=config.mask_filter_size, activation='relu', padding='causal')(mask_input)
    mask_dropout = MCDropout(rate=config.dropout_rate)(mask_conv)

    # Concatenate output from mask branch to main branch
    concat = Concatenate()([covariate_dropout, mask_dropout])

    # Second convolutional layer
    covariate_conv = Conv1D(filters=config.cov_filters,
                            kernel_size=config.cov_filter_size, activation='relu', padding='causal')(concat)
    covariate_dropout = MCDropout(rate=config.dropout_rate)(covariate_conv)
    
    mask_conv = Conv1D(filters=config.mask_filters,
                       kernel_size=config.mask_filter_size, activation='relu', padding='causal')(mask_conv)
    mask_dropout = MCDropout(rate=config.dropout_rate)(mask_conv)

    # Concatenate output from mask branch to main branch
    concat = Concatenate()([covariate_dropout, mask_dropout])

    # Dense layers
    flatten = Flatten()(concat)
    dense = Dense(units=config.dense_units, activation='relu')(flatten)
    dense_dropout = MCDropout(rate=config.dropout_rate)(dense)
    dense = Dense(units=config.dense_units, activation='relu')(dense_dropout)
    dense_dropout = MCDropout(rate=config.dropout_rate)(dense)

    # Define output layers based on specified prediction horizon
    output_layers = []
    for i in range(config.pred_horizon):
        output = Dense(units=2, activation='softmax')(dense_dropout)
        output_layers.append(output)

    # Construct and return model
    model = MatchNet(inputs=[covariate_input, mask_input], outputs=output_layers, config=config)

    return model
