from functional import *

class multihead_attention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads,**kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.projection_dim = embed_dim // num_heads
        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)
        self.combine_heads = tf.keras.layers.Dense(embed_dim)

    def attention_value(self, query, key, value):
        score=tf.matmul(query, key,transpose_b=True)
        dim_key=key.shape[-1]
        score/=np.sqrt(dim_key)
        weights=tf.keras.ops.softmax(score, axis=-1)
        output=tf.matmul(weights, value)
        return output

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def call(self, inputs):
        query, key, value = inputs
        batch_size = tf.shape(query)[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention = self.attention_value(query, key, value)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        output = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(output)
        return output

class encoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim,**kwargs):
        super().__init__(**kwargs)
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.ff_dim=ff_dim
        self.att = multihead_attention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1=tf.keras.layers.LayerNormalization()
        self.layernorm2=tf.keras.layers.LayerNormalization()
        
    def call(self, inputs):
        attn_output = self.att([inputs, inputs, inputs])
        add_norm1 = self.layernorm1(inputs + attn_output) #first add & normalize layer
        ffn_output = self.ffn(add_norm1)
        add_norm2=self.layernorm2(add_norm1 + ffn_output) # second add & normalize layer
        return add_norm2

class timeseries_transformer(funtional):
    def __init__(self,num_heads=8,embed_dim=256,ff_dim=64):
        super().__init__()
        self.num_heads=num_heads
        self.embed_dim=embed_dim
        self.ff_dim=ff_dim
    def positional_encoding(self,n=10000):
        p=np.zeros((self.seq_length,self.embed_dim))
        for k in range(self.seq_length):
            for i in np.arange(self.embed_dim//2):
                denominator=np.power(n, 2*i/self.embed_dim)
                p[k, 2*i]=np.sin(k/denominator)
                p[k, 2*i+1]=np.cos(k/denominator)
        return p
    def build(self):
        input_layer= tf.keras.layers.Input((self.seq_length,1))
        pos_encode_layer=self.positional_encoding()
        data=input_layer+pos_encode_layer
        encoder_layer=encoder(self.embed_dim,self.num_heads,self.ff_dim)(data)
        dense_layer=tf.keras.layers.Dense(500, activation="leaky_relu")(encoder_layer)
        dense_layer=tf.keras.layers.Dense(200, activation="leaky_relu")(dense_layer)
        dense_layer=tf.keras.layers.Dense(100, activation="leaky_relu")(dense_layer)
        dense_layer=tf.keras.layers.Dense(50, activation="leaky_relu")(dense_layer)
        dense_layer=tf.keras.layers.Dense(20, activation="leaky_relu")(dense_layer)
        flatten_layer=tf.keras.layers.Flatten()(dense_layer)
        output_layer=tf.keras.layers.Dense(1)(flatten_layer)
        model=tf.keras.Model(inputs=input_layer, outputs=output_layer)

        return model
    
    def train(self,epochs=10,optimizer='adam',loss='mae',batch_size=32):
        self.model=self.build()
        self.model.compile(optimizer=optimizer,loss=loss)
        best_lost=tf.keras.callbacks.ModelCheckpoint(os.path.join(Path().cwd(),f'models/transformer_{self.price_type}_{self.symbol}_{self.timeframe}.weights.h5'),save_weights_only=True,monitor='val_loss',mode='min',save_best_only=True)
        self.history=self.model.fit(self.x_train,self.y_train,epochs=epochs,batch_size=batch_size,callbacks=[best_lost],validation_data=(self.x_test,self.y_test))
        self.pred=self.model.predict(self.x).reshape(-1)
    
if __name__=='__main__':
    num_heads=8
    embed_dim=256
    ff_dim=64
    model=timeseries_transformer(num_heads,embed_dim,ff_dim)
    model.train(epochs=100)
    model.plot_summary()