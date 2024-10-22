from functional import *

class timeseries_rnn(funtional):
    def __init__(self):
        super().__init__()
        self.model_name='rnn'

    def build(self):
        input_layer=tf.keras.Input((self.seq_length,1))
        rnn_layer=tf.keras.layers.SimpleRNN(1)(input_layer)
        dense_layer=tf.keras.layers.Dense(500,activation='leaky_relu')(rnn_layer)
        normalization_layer=tf.keras.layers.LayerNormalization()(dense_layer)
        dense_layer=tf.keras.layers.Dense(200,activation='leaky_relu')(normalization_layer)
        dropout_layer=tf.keras.layers.Dropout(0.3)(dense_layer)
        dense_layer=tf.keras.layers.Dense(100,activation='leaky_relu')(dropout_layer)
        normalization_layer=tf.keras.layers.LayerNormalization()(dense_layer)
        dense_layer=tf.keras.layers.Dense(50,activation='leaky_relu')(normalization_layer)
        dropout_layer=tf.keras.layers.Dropout(0.3)(dense_layer)
        dense_layer=tf.keras.layers.Dense(20,activation='leaky_relu')(dropout_layer)
        normalization_layer=tf.keras.layers.LayerNormalization()(dense_layer)
        dense_layer=tf.keras.layers.Dense(10,activation='leaky_relu')(normalization_layer)
        dropout_layer=tf.keras.layers.Dropout(0.3)(dense_layer)
        output_layer=tf.keras.layers.Dense(1)(dropout_layer)
        model=tf.keras.Model(input_layer,output_layer)

        return model

if __name__=='__main__':
    model=timeseries_rnn()
    model.train(epochs=100,loss='mape')
    model.plot_summary()