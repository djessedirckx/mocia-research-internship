import keras_tuner as kt

from keras_tuner import HyperParameters
from tensorflow.keras.layers import Conv1D, Dense, Concatenate, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

from hyperparameter_tuning.RandomSearchConfig import RandomSearchConfig
from model.MatchNet import MatchNet
from model.MatchNetConfig import MatchNetConfig
from model.layers.MCDropout import MCDropout

class MatchNetHyperModel(kt.HyperModel):

    def __init__(self, search_config: RandomSearchConfig, matchnet_config: MatchNetConfig, name=None, tunable=True):
        super(MatchNetHyperModel, self).__init__(name=name, tunable=tunable)

        self.search_config: RandomSearchConfig = search_config
        self.matchnet_config: MatchNetConfig = matchnet_config

    def build(self, hp: HyperParameters):

        # Define input blocks
        covariate_input = Input(shape=self.search_config.cov_input_shape)
        mask_input = Input(shape=self.search_config.mask_input_shape)

        # Create the specified number of convolutional (parallel) streams
        x_covariate, x_mask = covariate_input, mask_input
        for _ in range(hp.Choice('conv_layers', self.search_config.conv_layers)):

            # Define hyperparameter selection ranges
            covariate_filters = hp.Choice('covariate_filters', self.search_config.covariate_filters)
            mask_filters = hp.Choice('mask_filters', self.search_config.mask_filters)
            conv_width = hp.Choice('conv_width', self.search_config.conv_filter_width)
            l1 = hp.Choice('l1', self.search_config.l1)
            l2 = hp.Choice('l2', self.search_config.l2)
            dropout_rate = hp.Choice('dropout_rate', self.search_config.dropout_rate)

            # Define layers
            x_covariate = Conv1D(filters=covariate_filters, kernel_size=conv_width, kernel_regularizer=l1_l2(l1, l2), activation='relu', padding='causal')(x_covariate)
            x_covariate = MCDropout(rate=dropout_rate)(x_covariate)    

            x_mask = Conv1D(filters=mask_filters, kernel_size=conv_width, kernel_regularizer=l1_l2(l1, l2), activation='relu', padding='causal')(x_mask)
            x_mask = MCDropout(rate=dropout_rate)(x_mask)

            # Concatenate output from mask branch to main branch
            x_covariate = Concatenate()([x_covariate, x_mask])

        # Dense layers
        x_covariate = Flatten()(x_covariate)

        # Create number of dense layers (based on hyperparameter choice)
        for _ in range(hp.Choice('dense_layers', self.search_config.connected_layers)):
            # Define hyperparameter selection range
            dense_units = hp.Choice('dense_units', self.search_config.connected_width)
        
            x_covariate = Dense(units=dense_units, activation='relu', kernel_regularizer=l1_l2(l1, l2))(x_covariate)
            x_covariate = MCDropout(dropout_rate)(x_covariate)

        # Define output layers based on specified prediction horizon
        output_layers = []
        for i in range(self.matchnet_config.pred_horizon):
            output = Dense(units=2, activation='softmax', kernel_regularizer=l1_l2(l1, l2))(x_covariate)
            output_layers.append(output)

        # Construct and return model
        model = MatchNet(inputs=[covariate_input, mask_input], outputs=output_layers, config=self.matchnet_config)
        optimizer = Adam(learning_rate=hp.Choice('lr_rate', self.search_config.lr))
        model.compile(optimizer)
        return model
