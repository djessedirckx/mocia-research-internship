from tensorflow.keras.layers import Dropout

class MCDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)