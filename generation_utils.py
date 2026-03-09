import tensorflow as tf

def top_k_top_p_logits(logits, k=0, p=1.0):
    logits = tf.cast(logits, tf.float32)
    if k > 0:
        values, _ = tf.math.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        logits = tf.where(logits < min_values, -1e10, logits)
    if p < 1.0:
        sorted_logits = tf.sort(logits, direction='DESCENDING')
        cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
        
        cutoff_index = tf.argmax(cumulative_probs > p, axis=-1)
        
        batch_size = tf.shape(logits)[0]
        gather_indices = tf.stack([tf.range(batch_size, dtype=cutoff_index.dtype), cutoff_index], axis=1)
        cutoff_values = tf.gather_nd(sorted_logits, gather_indices)
        
        logits = tf.where(logits < cutoff_values[:, tf.newaxis], -1e10, logits)
        
    return logits
