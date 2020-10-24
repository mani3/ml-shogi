import os
import glob
import pickle

from absl import logging as logger

import tensorflow as tf

import ml.dataset.kifu as kifu


class CSA(object):

  def __init__(
    self, train_dir, valid_dir,
    epochs, batch_size=64, seed=42):

    self.batch_size = batch_size
    self.seed = seed
    self.epochs = epochs

    self.train_dir = train_dir
    self.valid_dir = valid_dir

    self.train_list = glob.glob(os.path.join(train_dir, '*.pickle'))
    self.valid_list = glob.glob(os.path.join(valid_dir, '*.pickle'))

  def preprocess(self, filepath):
    x, y, win = tf.py_function(
      self.read_file, [filepath], [tf.float32, tf.int64, tf.int64])
    return x, y, win

  def read_file(self, filepath):
    filepath = filepath.numpy().decode("utf-8")
    with open(filepath, 'rb') as f:
      [x, y, win, move_number, steps] = pickle.load(f)
    return [x, y, win]

  def train_input_fn(self):
    batch_size = self.batch_size
    file_pattern = os.path.join(self.train_dir, '*.pickle')

    num_parallel = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.list_files(file_pattern, seed=self.seed)
    dataset = (dataset.shuffle(10000, seed=self.seed)
               .map(self.preprocess, num_parallel_calls=num_parallel)
               .repeat(self.epochs)
               .batch(batch_size)
               .prefetch(tf.data.experimental.AUTOTUNE))
    return dataset

  def valid_input_fn(self):
    batch_size = self.batch_size
    file_pattern = os.path.join(self.valid_dir, '*.pickle')

    num_parallel = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = (dataset.repeat(1)
               .map(self.preprocess, num_parallel_calls=num_parallel)
               .batch(batch_size)
               .cache()
               .prefetch(tf.data.experimental.AUTOTUNE))
    return dataset
