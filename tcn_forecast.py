from functional import *

class timeseries_tcn(funtional):
    def __init__(self,num_blocks=3,filters=32,kernel_size=3,dropout_rate=0.2):
        super().__init__()
        self.model_name='tcn'
        self.num_blocks=num_blocks
        self.filters=filters
        self.kernel_size=kernel_size
        self.dropout_rate=dropout_rate

    def residual_block(self,layer_input, dilation_rate, padding):
        shortcut=layer_input
        conv1d_layer=tf.keras.layers.Conv1D(self.filters,self.kernel_size, padding=padding, dilation_rate=dilation_rate)(layer_input)
        batch_normalize_layer=tf.keras.layers.BatchNormalization()(conv1d_layer)
        relu_layer=tf.keras.layers.ReLU()(batch_normalize_layer)
        dropout_layer= tf.keras.layers.SpatialDropout1D(self.dropout_rate)(relu_layer)
        conv1d_layer=tf.keras.layers.Conv1D(self.filters,self.kernel_size, padding=padding, dilation_rate=dilation_rate)(dropout_layer)
        batch_normalize_layer= tf.keras.layers.BatchNormalization()(conv1d_layer)
        
        if shortcut.shape[-1] != batch_normalize_layer.shape[-1]:
            shortcut=tf.keras.layers.Conv1D(self.filters, 1, padding='same')(shortcut)

        add_layer=tf.keras.layers.Add()([shortcut, batch_normalize_layer])
        relu_layer=tf.keras.layers.ReLU()(add_layer)
        return relu_layer

    def build(self):
        input_layer=tf.keras.Input((self.seq_length,1))
        residual_block_layer=self.residual_block(input_layer,1,padding='causal')
        for i in range(1,self.num_blocks):
            dilation_rate = 2 ** i 
            residual_block_layer=self.residual_block(residual_block_layer,dilation_rate,padding='causal')

        conv1d_layer=tf.keras.layers.Conv1D(1,1)(residual_block_layer)
        squueze_layer=tf.keras.layers.Lambda(lambda k: tf.squeeze(k, axis=-1))(conv1d_layer)
        output_layer=tf.keras.layers.Dense(1)(squueze_layer)
        model = tf.keras.Model(input_layer, output_layer)
        return model

if __name__=='__main__':
    model=timeseries_tcn()
    model.train(epochs=100,loss='mape')
    model.plot_summary()