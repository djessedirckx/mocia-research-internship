from tensorflow.keras.layers import Conv1D, Dense, Concatenate, Input, Flatten
from tensorflow.keras.regularizers import l1_l2

from model.MatchNet import MatchNet
from model.MatchNetConfig import MatchNetConfig
from model.layers.MCDropout import MCDropout


def build_model(config: MatchNetConfig) -> MatchNet:

    # Define input blocks
    covariate_input = Input(shape=config.cov_input_shape)
    mask_input = Input(shape=config.mask_input_shape)

    # Create the specified number of convolutional (parallel) streams
    x_covariate, x_mask = covariate_input, mask_input
    for _ in range(config.conv_blocks):
        x_covariate = Conv1D(filters=config.cov_filters, kernel_size=config.cov_filter_size, kernel_regularizer=l1_l2(config.l1, config.l2), activation='relu', padding='causal')(x_covariate)
        x_covariate = MCDropout(rate=config.dropout_rate)(x_covariate)    

        x_mask = Conv1D(filters=config.mask_filters, kernel_size=config.mask_filter_size, kernel_regularizer=l1_l2(config.l1, config.l2), activation='relu', padding='causal')(x_mask)
        x_mask = MCDropout(rate=config.dropout_rate)(x_mask)

        # Concatenate output from mask branch to main branch
        x_covariate = Concatenate()([x_covariate, x_mask])

    # Dense layers
    x_covariate = Flatten()(x_covariate)

    # Create the specified number of dense layers
    for _ in range(config.dense_layers):
        x_covariate = Dense(units=config.dense_units, activation='relu', kernel_regularizer=l1_l2(config.l1, config.l2))(x_covariate)
        x_covariate = MCDropout(rate=config.dropout_rate)(x_covariate)

    # Define output layers based on specified prediction horizon
    output_layers = []
    for i in range(config.pred_horizon):
        output = Dense(units=2, activation='softmax', kernel_regularizer=l1_l2(config.l1, config.l2))(x_covariate)
        output_layers.append(output)

    # Construct and return model
    model = MatchNet(inputs=[covariate_input, mask_input], outputs=output_layers, config=config)

    return model
