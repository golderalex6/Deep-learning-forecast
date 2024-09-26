from functional import *

class timeseries_lstm(funtional):
    def __init__(self):
        super().__init__()
    def build(self):

        input_layer=tf.keras.Input((self.seq_length,1))
        lstm_layer=tf.keras.layers.LSTM(1)(input_layer)
        dense_layer=tf.keras.layers.Dense(500,activation='leaky_relu')(lstm_layer)
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

    def train(self,epochs=10,optimizer='adam',loss='mae',batch_size=32):
        self.model=self.build()
        self.model.compile(optimizer=optimizer,loss=loss)
        best_lost=tf.keras.callbacks.ModelCheckpoint(os.path.join(Path().cwd(),f'models/lstm_{self.price_type}_{self.symbol}_{self.timeframe}.weights.h5'),save_weights_only=True,monitor='loss',mode='min',save_best_only=True)
        self.history=self.model.fit(self.x_train,self.y_train,epochs=epochs,batch_size=batch_size,callbacks=[best_lost],validation_data=(self.x_test,self.y_test))
        self.pred=self.model.predict(self.x).reshape(-1)

if __name__=='__main__':
    model=timeseries_lstm()
    model.train(epochs=100,loss='mape')
    model.plot_summary()