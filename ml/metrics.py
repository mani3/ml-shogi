import numpy as np
import tensorflow as tf


class Metrics(object):

  def __init__(self, loss_types=['policy', 'values']):
    m = tf.keras.metrics

    self.metrics_loss_train = []
    self.metrics_loss_valid = []

    for t in loss_types:
      name = 'Loss/{}/train'.format(t)
      l = m.Mean(name, dtype=tf.float32)
      self.metrics_loss_train.append(l)

      name = 'Loss/{}/valid'.format(t)
      l = m.Mean(name, dtype=tf.float32)
      self.metrics_loss_valid.append(l)

    self.metrics_accuracy_train = []
    self.metrics_accuracy_valid = []

    for t in loss_types:
      name = 'Accuracy/{}/train'.format(t)
      l = m.SparseCategoricalAccuracy(name, dtype=tf.float32)
      self.metrics_accuracy_train.append(l)

      name = 'Accuracy/{}/valid'.format(t)
      l = m.BinaryCrossentropy(name, dtype=tf.float32)
      self.metrics_accuracy_valid.append(l)

  def set_train_loss(self, logs: list):
    for m, l in zip(self.metrics_loss_train, logs):
      m(l)

  def set_train_accuracy(self, y_true: list, y_pred: list):
    for m, t, p in zip(self.metrics_accuracy_train, y_true, y_pred):
      m(np.expand_dims(t, axis=1), p)

  def write_train(self, step):
    for m in self.metrics_loss_train:
      tf.summary.scalar(m.name, m.result(), step=step)
    for m in self.metrics_accuracy_train:
      tf.summary.scalar(m.name, m.result(), step=step)

  def reset_train(self):
    for m in self.metrics_loss_train:
      m.reset_states()
    for m in self.metrics_accuracy_train:
      m.reset_states()

  def set_valid_loss(self, logs: list):
    for m, l in zip(self.metrics_loss_valid, logs):
      m(l)

  def set_valid_accuracy(self, y_true: list, y_pred: list):
    for m, t, p in zip(self.metrics_accuracy_valid, y_true, y_pred):
      m(np.expand_dims(t, axis=1), p)

  def get_valid_accuracy(self):
    return [m.result() for m in self.metrics_accuracy_valid]

  def write_valid(self, step):
    for m in self.metrics_loss_valid:
      tf.summary.scalar(m.name, m.result(), step=step)
    for m in self.metrics_accuracy_valid:
      tf.summary.scalar(m.name, m.result(), step=step)

  def reset_valid(self):
    for m in self.metrics_loss_valid:
      m.reset_states()
    for m in self.metrics_accuracy_valid:
      m.reset_states()
