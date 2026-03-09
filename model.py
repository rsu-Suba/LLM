import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import yaml
import os

def register_serializable(cls):
    return tf.keras.utils.register_keras_serializable(package="Custom")(cls)

def std_init():
    return tf.keras.initializers.TruncatedNormal(stddev=0.002)

@register_serializable
class RMSNorm(layers.Layer):
    def __init__(self, dim, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = epsilon
        self.dim = dim
        self.scale = self.add_weight(
            name="scale",
            shape=(dim,),
            initializer="ones",
            trainable=True
        )

    def call(self, x):
        x_dtype = x.dtype
        x_f32 = tf.cast(x, tf.float32)
        variance = tf.reduce_mean(tf.square(x_f32), axis=-1, keepdims=True)
        norm_x = x_f32 * tf.math.rsqrt(variance + self.eps)
        return tf.cast(norm_x, x_dtype) * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "epsilon": self.eps})
        return config

@register_serializable
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, embeddings_initializer=std_init())
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim, embeddings_initializer=std_init())
        self.norm = RMSNorm(embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return self.norm(x + positions)

    def get_config(self):
        config = super().get_config()
        config.update({"maxlen": self.maxlen, "vocab_size": self.vocab_size, "embed_dim": self.embed_dim})
        return config

@register_serializable
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rate = rate
        ff_dim = embed_dim * 4
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads,
            kernel_initializer=std_init()
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation=tf.nn.gelu, kernel_initializer=std_init()),
            layers.Dense(embed_dim, kernel_initializer=std_init()),
        ])
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        causal_mask = tf.cast(causal_mask, tf.bool)
        
        x = self.norm1(inputs)
        attn_output = self.att(query=x, value=x, key=x, attention_mask=causal_mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = inputs + attn_output
        
        normed_x = self.norm2(x)
        ffn_output = self.ffn(normed_x)
        ffn_output = self.dropout2(ffn_output, training=training)
        return x + ffn_output

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "rate": self.rate})
        return config

@register_serializable
class TiedOutput(layers.Layer):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size

    def build(self, input_shape):
        self.bias = self.add_weight(name="bias", shape=(self.vocab_size,), initializer="zeros", trainable=True)

    def call(self, inputs, embedding_weights=None):
        logits = tf.matmul(inputs, embedding_weights, transpose_b=True)
        return logits + self.bias

    def get_config(self):
        config = super().get_config()
        config.update({"vocab_size": self.vocab_size})
        return config

@register_serializable
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_learning_rate, warmup_steps, total_steps, end_learning_rate, **kwargs):
        super().__init__()
        self.peak_learning_rate = peak_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.end_learning_rate = end_learning_rate
        self.decay_steps = total_steps - warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        def warmup_fn(): return self.peak_learning_rate * (step / tf.cast(self.warmup_steps, tf.float32))
        def decay_fn():
            step_in_decay = step - self.warmup_steps
            cosine_decay = 0.5 * (1 + tf.cos(tf.constant(math.pi) * step_in_decay / tf.cast(self.decay_steps, tf.float32)))
            return (self.peak_learning_rate - self.end_learning_rate) * cosine_decay + self.end_learning_rate
        return tf.cond(step < self.warmup_steps, warmup_fn, decay_fn)

    def get_config(self):
        return {
            "peak_learning_rate": self.peak_learning_rate,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "end_learning_rate": self.end_learning_rate,
        }

def build_model(vocab_size, max_len, embed_dim, num_transformer_blocks, num_heads):
    inputs = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
    x = embedding_layer(inputs)

    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads)(x)

    x = RMSNorm(embed_dim)(x)
    outputs = TiedOutput(vocab_size=vocab_size)(x, embedding_weights=embedding_layer.token_emb.embeddings)
    outputs = layers.Activation("linear", dtype="float32")(outputs)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == '__main__':
    yaml_path = "model_param.yaml"
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        model_name = config['default_model']
        p = config[model_name]
        print(f"--- Model Spec Loaded: {model_name} ---")
        model = build_model(p['VOCAB_SIZE'], p['MAX_LEN'], p['EMBED_DIM'], p['NUM_TRANSFORMER_BLOCKS'], p['NUM_HEADS'])
        model.summary()
