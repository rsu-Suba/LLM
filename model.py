import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        ff_dim = embed_dim * 4
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        input_shape = tf.shape(inputs)
        seq_len = input_shape[1]
        causal_mask = self.create_causal_mask(seq_len)

        # Multi-Head Attention
        normed_inputs = self.layernorm1(tf.cast(inputs, tf.float32))
        attn_output = self.att(
            query=normed_inputs,
            value=normed_inputs,
            key=normed_inputs,
            attention_mask=causal_mask,
            training=training
        )
        attn_output = tf.cast(attn_output, inputs.dtype)
        attn_output = self.dropout1(attn_output, training=training)

        x = inputs + attn_output

        # Feed Forward
        normed_x = self.layernorm2(tf.cast(x, tf.float32))
        ffn_output = self.ffn(normed_x)
        ffn_output = tf.cast(ffn_output, inputs.dtype)
        ffn_output = self.dropout2(ffn_output, training=training)

        return x + ffn_output

    def create_causal_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return tf.cast(mask, tf.bool)

class TiedOutputDense(layers.Layer):
    def __init__(self, token_embedding_layer, **kwargs):
        super().__init__(**kwargs)
        self.token_embedding_layer = token_embedding_layer

    def build(self, input_shape):
        self.bias = self.add_weight(
            shape=(self.token_embedding_layer.input_dim,),
            initializer="zeros",
            trainable=True,
            name="output_bias"
        )

    def call(self, inputs):
        output = tf.matmul(inputs, self.token_embedding_layer.embeddings, transpose_b=True)
        return output + self.bias

    def get_config(self):
        config = super().get_config()
        config.update({
            "token_embedding_layer": tf.keras.layers.serialize(self.token_embedding_layer)
        })
        return config

    @classmethod
    def from_config(cls, config):
        embedding_layer_config = config.pop("token_embedding_layer")
        token_embedding_layer = tf.keras.layers.deserialize(embedding_layer_config)
        return cls(token_embedding_layer, **config)


def build_model(vocab_size, max_len, embed_dim, num_transformer_blocks, num_heads):
    inputs = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
    x = embedding_layer(inputs)

    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads)(x)

    outputs = TiedOutputDense(token_embedding_layer=embedding_layer.token_emb)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

if __name__ == '__main__':
    llm = build_model()
    llm.summary()