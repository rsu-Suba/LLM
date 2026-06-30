import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import yaml
import os
import argparse

def register_serializable(cls):
    return tf.keras.utils.register_keras_serializable(package="Custom")(cls)

def std_init():
    return tf.keras.initializers.TruncatedNormal(stddev=0.02)

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

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "epsilon": self.eps})
        return config

@register_serializable
class TokenEmbedding(layers.Layer):
    def __init__(self, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, embeddings_initializer=std_init())
        self.norm = RMSNorm(embed_dim)

    def call(self, x):
        x = self.token_emb(x)
        return self.norm(x)

    def get_config(self):
        config = super().get_config()
        config.update({"vocab_size": self.vocab_size, "embed_dim": self.embed_dim})
        return config

@register_serializable
class RoPEMultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, head_dim, num_kv_heads=None, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_queries_per_kv = num_heads // self.num_kv_heads
        self.scale = head_dim ** -0.5

        self.q_proj = layers.Dense(num_heads * head_dim, use_bias=False, kernel_initializer=std_init())
        self.k_proj = layers.Dense(self.num_kv_heads * head_dim, use_bias=False, kernel_initializer=std_init())
        self.v_proj = layers.Dense(self.num_kv_heads * head_dim, use_bias=False, kernel_initializer=std_init())
        self.o_proj = layers.Dense(num_heads * head_dim, use_bias=False, kernel_initializer=std_init())

    def build(self, input_shape):
        self.q_proj.build(input_shape)
        self.k_proj.build(input_shape)
        self.v_proj.build(input_shape)
        self.o_proj.build(input_shape[:-1] + (self.num_heads * self.head_dim,))
        self.built = True

    def _apply_rope(self, x, seq_len):
        pos = tf.cast(tf.range(seq_len), dtype=tf.float32)
        indices = tf.cast(tf.range(0, self.head_dim, 2), dtype=tf.float32)
        freqs = 1.0 / (10000.0 ** (indices / self.head_dim))

        angles = pos[:, None] * freqs[None, :]
        cos = tf.cos(angles)[None, :, None, :]
        sin = tf.sin(angles)[None, :, None, :]

        cos = tf.cast(cos, x.dtype)
        sin = tf.cast(sin, x.dtype)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos

        res = tf.stack([rx1, rx2], axis=-1)
        return tf.reshape(res, tf.shape(x))

    def _repeat_kv(self, x, n_rep):
        if n_rep == 1:
            return x
        # x shape: (B, S, n_kv, D)
        x = tf.expand_dims(x, axis=3)  # (B, S, n_kv, 1, D)
        x = tf.tile(x, [1, 1, 1, n_rep, 1])  # (B, S, n_kv, n_rep, D)
        return tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], self.num_heads, self.head_dim))

    def call(self, x, attention_mask=None, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = tf.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = tf.reshape(k, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        v = tf.reshape(v, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        q = self._apply_rope(q, seq_len)
        k = self._apply_rope(k, seq_len)

        # GQA: Repeat Key/Value for each Query head
        k = self._repeat_kv(k, self.num_queries_per_kv)
        v = self._repeat_kv(v, self.num_queries_per_kv)

        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 3, 1])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        scores = tf.matmul(q, k) * self.scale

        if attention_mask is not None:
            mask = tf.cast(attention_mask, scores.dtype)
            scores += (1.0 - mask) * -1e4

        attn = tf.nn.softmax(tf.cast(scores, tf.float32), axis=-1)
        attn = tf.cast(attn, scores.dtype)

        out = tf.matmul(attn, v)

        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, seq_len, self.num_heads * self.head_dim))

        return self.o_proj(out)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.num_heads * self.head_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "num_kv_heads": self.num_kv_heads
        })
        return config

@register_serializable
class SwiGLU(layers.Layer):
    def __init__(self, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.ff_dim = ff_dim
        self.w1 = layers.Dense(ff_dim, use_bias=False, kernel_initializer=std_init())
        self.w2 = layers.Dense(ff_dim, use_bias=False, kernel_initializer=std_init())

    def build(self, input_shape):
        self.w1.build(input_shape)
        self.w2.build(input_shape)
        self.built = True

    def call(self, x):
        return tf.nn.silu(self.w1(x)) * self.w2(x)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.ff_dim,)

    def get_config(self):
        config = super().get_config()
        config.update({"ff_dim": self.ff_dim})
        return config

@register_serializable
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, num_kv_heads=None, ff_dim=None, rate=0.03, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.ff_dim = ff_dim if ff_dim is not None else int(embed_dim * 8 / 3)
        self.rate = rate
        head_dim = embed_dim // num_heads

        self.att = RoPEMultiHeadAttention(num_heads=num_heads, head_dim=head_dim, num_kv_heads=self.num_kv_heads)

        # SwiGLU Configuration
        self.ffn = SwiGLU(self.ff_dim)
        self.wo = layers.Dense(embed_dim, use_bias=False, kernel_initializer=std_init())

        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def build(self, input_shape):
        self.norm1.build(input_shape)
        self.att.build(input_shape)
        self.norm2.build(input_shape)
        self.ffn.build(input_shape)
        self.wo.build(self.ffn.compute_output_shape(input_shape))
        self.built = True

    def call(self, inputs, training=False):
        def _compute(x):
            seq_len = tf.shape(x)[1]
            causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            causal_mask = causal_mask[None, None, :, :]

            normed = self.norm1(x)
            attn_output = self.att(normed, attention_mask=causal_mask, training=training)
            attn_output = self.dropout1(attn_output, training=training)
            x_out = x + attn_output

            normed2 = self.norm2(x_out)
            ffn_output = self.ffn(normed2)
            ffn_output = self.wo(ffn_output)
            ffn_output = self.dropout2(ffn_output, training=training)
            return x_out + ffn_output

        if training:
            return tf.recompute_grad(_compute)(inputs)
        return _compute(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
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

def build_model(vocab_size, max_len, embed_dim, num_transformer_blocks, num_heads, num_kv_heads=None, ff_dim=None):
    inputs = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding_layer = TokenEmbedding(vocab_size, embed_dim)
    x = embedding_layer(inputs)

    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads, num_kv_heads=num_kv_heads, ff_dim=ff_dim)(x)

    x = RMSNorm(embed_dim)(x)
    outputs = TiedOutput(vocab_size=vocab_size)(x, embedding_weights=embedding_layer.token_emb.embeddings)
    outputs = layers.Activation("linear", dtype="float32")(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='default', help='Name of the model config to use')
    args = parser.parse_args()

    yaml_path = "model_param.yaml"
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        model_name = config['default_model'] if args.model == 'default' else args.model
        if model_name in config:
            p = config[model_name]
            print(f"--- Model Summary: {model_name} ---")
            num_kv = p.get('NUM_KV_HEADS', None)
            ff_dim = p.get('FF_DIM', None)
            model = build_model(p['VOCAB_SIZE'], p['MAX_LEN'], p['EMBED_DIM'], p['NUM_TRANSFORMER_BLOCKS'], p['NUM_HEADS'], num_kv_heads=num_kv, ff_dim=ff_dim)
            model.summary()
        else:
            print(f"Error: Model config '{model_name}' not found in {yaml_path}")
