import absl
import tensorflow as tf

logger = absl.logging


class CSATFRecord(object):

  feature_description = {
    'feature': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'channel': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'move_label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'win': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'move_number': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'steps': tf.io.FixedLenFeature([], tf.int64, default_value=0),
  }

  def __init__(
    self, train_tfrecord, valid_tfrecord,
    epochs, batch_size=64, seed=42):

    self.batch_size = batch_size
    self.seed = seed
    self.epochs = epochs

    self.train_tfrecord = train_tfrecord
    self.valid_tfrecord = valid_tfrecord

    self.train_size = self.get_size(train_tfrecord)
    self.valid_size = self.get_size(valid_tfrecord)

    logger.info(f'train_count: {self.train_size}, valid_count: {self.valid_size}')

  def get_size(self, filename):
    return sum(1 for _ in tf.data.TFRecordDataset(filename))

  def preprocess(self, example):
    example = tf.io.parse_single_example(example, self.feature_description)
    x = tf.io.decode_raw(example['feature'], tf.uint8)
    x = tf.reshape(x, [9, 9, 104])
    x = tf.cast(x, tf.float32)
    return x, example['move_label'], example['win']

  def train_input_fn(self):
    batch_size = self.batch_size

    num_parallel = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset([self.train_tfrecord])
    dataset = (dataset.shuffle(10000, seed=self.seed)
               .map(self.preprocess, num_parallel_calls=num_parallel)
               .repeat(self.epochs)
               .batch(batch_size)
               .prefetch(tf.data.experimental.AUTOTUNE))
    return dataset

  def valid_input_fn(self):
    batch_size = self.batch_size

    num_parallel = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset([self.valid_tfrecord])
    dataset = (dataset.repeat(1)
               .map(self.preprocess, num_parallel_calls=num_parallel)
               .batch(batch_size)
               .prefetch(tf.data.experimental.AUTOTUNE))
    return dataset
