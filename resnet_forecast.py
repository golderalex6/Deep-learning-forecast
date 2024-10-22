from functional import *

class timeseries_resnet(funtional):
    def __init__(self,num_blocks=3, filters=32, kernel_size=3):
        super().__init__()
        self.model_name='resnet'
        self.num_blocks=num_blocks
        self.filters=filters
        self.kernel_size=kernel_size

    def residual_block(self,layer_input):
        shortcut=layer_input
        conv1d_layer=tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='same')(layer_input)
        batch_normalize_layer=tf.keras.layers.BatchNormalization()(conv1d_layer)
        relu_layer=tf.keras.layers.ReLU()(batch_normalize_layer)
        conv1d_layer=tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='same')(relu_layer)
        batch_normalize_layer=tf.keras.layers.BatchNormalization()(conv1d_layer)
        add_layer=tf.keras.layers.Add()([batch_normalize_layer, shortcut])
        relu_layer=tf.keras.layers.ReLU()(add_layer)
        return relu_layer

    def build(self):

        input_layer=tf.keras.Input((self.seq_length,1))
        conv1d_layer=tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='same')(input_layer)
        batch_normalize_layer= tf.keras.layers.BatchNormalization()(conv1d_layer)
        relu_layer=tf.keras.layers.ReLU()(batch_normalize_layer)
        
        
        residual_block_layer=self.residual_block(relu_layer)
        for _ in range(self.num_blocks-1):
            residual_block_layer=self.residual_block(residual_block_layer)
        
        average_pooling_layer=tf.keras.layers.GlobalAveragePooling1D()(residual_block_layer)

        #EURUSD
        # output_layer=tf.keras.layers.Dense(1)(average_pooling_layer)

        #USDVND
        average_pooling_layer=tf.keras.layers.GlobalAveragePooling1D()(residual_block_layer)
        dense_layer=tf.keras.layers.Dense(50,activation='leaky_relu')(average_pooling_layer)
        dense_layer=tf.keras.layers.Dense(20,activation='leaky_relu')(dense_layer)
        dense_layer=tf.keras.layers.Dense(10,activation='leaky_relu')(dense_layer)
        dense_layer=tf.keras.layers.Dense(5,activation='leaky_relu')(dense_layer)
        output_layer=tf.keras.layers.Dense(1)(dense_layer)

        model = tf.keras.Model(input_layer, outputs=output_layer)
        return model
if __name__=='__main__':
    model=timeseries_resnet()
    model.train(epochs=300,loss='mape')
    model.plot_summary()