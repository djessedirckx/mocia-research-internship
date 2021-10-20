from typing import Dict

from keras_tuner import HyperParameters
from tensorflow.keras.layers import Conv1D, Dense, Concatenate, Input, Flatten
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.models import Model

from model.MatchNet import MatchNet
from model.MatchNetConfig import MatchNetConfig
from model.layers.MCDropout import MCDropout

def build_model_for_search(hp: HyperParameters):

    # Choose hyperparameters
    window_size = hp.Choice('shape', [1, 2, 3])
    pred_horizon = hp.Choice('pred_horizon', [1, 2, 3])

    # Define input blocks
    covariate_input = Input(shape=(window_size, 35))
    x_covariate = Dense(32, activation='relu')(covariate_input)

    output_layers = []
    for i in range(pred_horizon):
        output = Dense(units=2, activation='softmax')(x_covariate)
        output_layers.append(output)

    # Construct and return model
    model = Model(inputs=covariate_input, outputs=output_layers)

    return model