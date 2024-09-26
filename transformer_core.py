import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
plt.rc('figure',titleweight='bold',titlesize='large',figsize=(15,6))
plt.rc('axes',titleweight='bold',titlesize='large',labelweight='bold',labelsize='large',grid=True)

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