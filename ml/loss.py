import tensorflow as tf


def simple_loss():
  cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()
  binary_entropy = tf.keras.losses.BinaryCrossentropy()

  def loss(y_true: list, y_pred: list, tag='train'):
    losses = [
      cross_entropy(y_true[0], y_pred[0]),
      binary_entropy(y_true[1], y_pred[1])
    ]
    return losses
  return loss
