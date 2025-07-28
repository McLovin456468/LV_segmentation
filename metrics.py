import tensorflow as tf


def compute_soft_dice(y_actual, y_output, mode='jaccard', epsilon=0.1):

    y_actual_flat = tf.reshape(tf.cast(y_actual, tf.float32), [-1])
    y_output_flat = tf.reshape(tf.cast(y_output, tf.float32), [-1])

    overlap = tf.reduce_sum(y_actual_flat * y_output_flat)

    if mode == 'jaccard':
        total = tf.reduce_sum(tf.square(y_actual_flat)) + tf.reduce_sum(tf.square(y_output_flat))
    elif mode == 'sorensen':
        total = tf.reduce_sum(y_actual_flat) + tf.reduce_sum(y_output_flat)
    else:
        raise ValueError(f"Недопустимый параметр 'mode': {mode}")
    return (2.0 * overlap + epsilon) / (total + epsilon)


def dice_loss_metric(y_actual, y_output):
    return 1.0 - compute_soft_dice(y_actual, y_output, epsilon=0.01)
