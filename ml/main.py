import os
import datetime

import absl
from absl import app
from absl import flags

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from ml.loss import simple_loss
from ml.model import (
  cnn,
  resnet,
)

from ml.trainer import Trainer
from ml.dataset.csa_tfrecord import CSATFRecord
from ml.dataset.common import MOVE_DIRECTION_LABEL_NUM


logger = absl.logging
FLAGS = flags.FLAGS

flags.DEFINE_string('train_tfrecord', './dataset/tfrecords/train.tfrecord', 'Train tfrecord')
flags.DEFINE_string('valid_tfrecord', './dataset/tfrecords/valid.tfrecord', 'Valid tfrecord')
flags.DEFINE_string('logdir', None, 'Log directory')

flags.DEFINE_string('model_name', 'cnn_simple192', 'cnn')
flags.DEFINE_string('loss_name', '', '')
flags.DEFINE_string('optimizer_name', 'adam', 'rmsprop, adam')
flags.DEFINE_string('saved_model_path', None, 'load weights path')

flags.DEFINE_string('activation_name', 'leaky_relu', 'activation name')

flags.DEFINE_integer('image_size', 9, 'Image size')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('epochs', 30, 'Epochs')

flags.DEFINE_integer('logging_steps', None, 'Logging steps')
flags.DEFINE_integer('seed', 1234, 'Seed')

flags.DEFINE_integer('reduce_patience', 16, 'Reduce patience: 16')

flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate')
flags.DEFINE_float('lr_decay_steps', 2250, 'Learning rate')
flags.DEFINE_float('lr_decay_rate', 1.0, 'Learning rate')

flags.DEFINE_float('focal_gamma', 2.0, 'focal loss gamma')
flags.DEFINE_float('focal_alpha', 0.25, 'focal loss alpha')
flags.DEFINE_float('mobilenet_alpha', 1.0, 'mobilenet alpha')

flags.DEFINE_float('reduce_factor', np.sqrt(0.1), 'Reduce factor: 0.0 ~ 1.0')
flags.DEFINE_float('reduce_min_lr', 1e-6, 'Reduce min lr')
flags.DEFINE_boolean('exponential_decay', False, 'Enable exponential decay')


def get_optimizer(name):
  if FLAGS.exponential_decay:
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=FLAGS.learning_rate,
      decay_steps=FLAGS.lr_decay_steps,
      decay_rate=FLAGS.lr_decay_rate,
      staircase=True)
  else:
    lr = FLAGS.learning_rate

  if name == 'rmsprop':
    return tf.keras.optimizers.RMSprop(learning_rate=lr)
  elif name == 'adam':
    return tf.keras.optimizers.Adam(learning_rate=lr)
  elif name == 'sgd':
    return tf.keras.optimizers.SGD(
      learning_rate=lr, momentum=0.9, nesterov=True)
  elif name == 'radam':
    return tfa.optimizers.RectifiedAdam(
      learning_rate=lr, total_steps=FLAGS.total_steps,
      warmup_proportion=0.1, min_lr=FLAGS.reduce_min_lr)
  else:
    return tf.keras.optimizers.Adam(learning_rate=lr)


def get_model(name):
  if name == 'cnn_simple192':
    return cnn.simple192
  elif name == 'resnet_20':
    return resnet.ResNet20
  elif name == 'resnet_40':
    return resnet.ResNet40
  else:
    raise ValueError('{} is not support model_name'.format(name))


def get_loss(name):
  return simple_loss()


def run(logdir, model_name, opt_name, loss_name):
  logger.info(f'model_name: {model_name}')
  logger.info(f'opt_name: {opt_name}')

  model = get_model(model_name)(
    input_shape=(9, 9, 104), classes=MOVE_DIRECTION_LABEL_NUM)
  logger.info(model.summary())

  if FLAGS.saved_model_path:
    model = tf.keras.models.load_model(FLAGS.saved_model_path)

  optimizer = get_optimizer(opt_name)
  loss = get_loss(loss_name)

  trainer = Trainer(model, logdir, optimizer, loss)

  with trainer.summary_writer.as_default():
    tf.summary.text('parameters', FLAGS.flags_into_string(), step=0)

  dataset = CSATFRecord(
    FLAGS.train_tfrecord, FLAGS.valid_tfrecord, FLAGS.epochs, FLAGS.batch_size)

  # 3143460
  if FLAGS.logging_steps:
    logging_steps = FLAGS.logging_steps
  else:
    logging_steps = dataset.train_size // FLAGS.batch_size
  logger.info(f'logging_steps: {logging_steps}')

  trainer.train(dataset, logging_steps, FLAGS.learning_rate)
  trainer.save()


def main(args):
  logger.info(FLAGS.flags_into_string())
  if FLAGS.logdir:
    logdir = FLAGS.logdir
  else:
    logdir = os.path.join(
      './logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
  run(logdir, FLAGS.model_name, FLAGS.optimizer_name, FLAGS.loss_name)


if __name__ == "__main__":
  app.run(main)


# kifu_list_train_path = 'kifulist_train_100.txt'
# kifu_list_valid_path = 'kifulist_valid_100.txt'
# csa = CSA(kifu_list_train_path, kifu_list_valid_path, 1)

# for x, y1, y2 in csa.train_input_fn().take(1):
#   print(f'feature: {x.shape}, {x.dtype}, label: {y1}, {y2}')

# for x, y1, y2 in csa.valid_input_fn().take(1):
#   print(f'feature: {x.shape}, {x.dtype}, label: {y1}, {y2}')

