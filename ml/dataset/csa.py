import os
import time
import pickle

from absl import logging as logger

import tensorflow as tf

import ml.dataset.kifu as kifu


class CSA(object):

  def __init__(
    self, kifu_list_train_path, kifu_list_valid_path,
    epochs, batch_size=64, seed=42):

    self.batch_size = batch_size
    self.seed = seed
    self.epochs = epochs
    train_path, _ = os.path.splitext(kifu_list_train_path)
    train_pickle_path = train_path + '.pickle'

    valid_path, _ = os.path.splitext(kifu_list_valid_path)
    valid_pickle_path = valid_path + '.pickle'

    # Train data
    start_time = time.time()
    logger.info(f'Start to load train kifu list: {kifu_list_train_path}')

    if os.path.exists(train_pickle_path):
      with open(train_pickle_path, 'rb') as f:
        self.positions_train = pickle.load(f)
    else:  
      self.positions_train = kifu.read(kifu_list_train_path)
      with open(train_pickle_path, 'wb') as f:
        pickle.dump(self.positions_train, f, pickle.HIGHEST_PROTOCOL)

    logger.info(f'End to load train kifu list: {time.time() - start_time} s')

    # Valid data
    start_time = time.time()
    logger.info(f'Start to load valid kifu list: {kifu_list_valid_path}')      

    if os.path.exists(valid_pickle_path):
      with open(valid_pickle_path, 'rb') as f:
        self.positions_valid = pickle.load(f)
    else:
      self.positions_valid = kifu.read(kifu_list_valid_path)
      with open(valid_pickle_path, 'wb') as f:
        pickle.dump(self.positions_valid, f, pickle.HIGHEST_PROTOCOL)

    logger.info(f'End to load valid kifu list: {time.time() - start_time} s')

    logger.info(f'position_train: {len(self.positions_train[0])}')
    logger.info(f'position_valid: {len(self.positions_valid[0])}')

  def preprocess(self, x, y1, y2):
    x = tf.cast(x, dtype=tf.float32)
    return x, y1, y2

  def train_input_fn(self):
    batch_size = self.batch_size
    feature_np, move_np, win_np = self.positions_train

    num_call = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices((feature_np, move_np, win_np))
    dataset = (dataset.shuffle(10000, seed=self.seed)
               .repeat(self.epochs)
               .map(self.preprocess, num_parallel_calls=num_call)
               .batch(batch_size)
               .prefetch(tf.data.experimental.AUTOTUNE))
    return dataset

  def valid_input_fn(self):
    batch_size = self.batch_size
    feature_np, move_np, win_np = self.positions_valid

    num_parallel = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices((feature_np, move_np, win_np))
    dataset = (dataset.repeat(1)
               .batch(batch_size)
               .map(self.preprocess, num_parallel_calls=num_parallel)
               .cache()
               .prefetch(tf.data.experimental.AUTOTUNE))
    return dataset
