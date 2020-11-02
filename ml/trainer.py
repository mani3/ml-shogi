import os
import time
import datetime

import absl

import sklearn
import numpy as np
import tensorflow as tf


from ml.metrics import Metrics
from ml.reducer import ReduceLearningRate

from ml.dataset.common import MOVE_DIRECTION_LABEL_NUM

logger = absl.logging


class Trainer(object):

  def __init__(
    self, model, logdir, optimizer, loss_fn,
    classes=9 * 9 * MOVE_DIRECTION_LABEL_NUM,
    labels=['root', 'vowel', 'consonant'],
    factor=np.sqrt(0.1),
    patience=16,
    min_lr=1e-10,
    alpha=None,
    only_roots=False):

    # model
    self.model = model
    self.logdir = logdir
    self.logging_steps = 1000
    self.classes = classes
    self.labels = labels
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.best_threshold = 0.97
    self.alpha = alpha
    self.only_roots = only_roots

    self.callbacks = self.get_callbacks(logdir)
    for callback in self.callbacks:
      callback.set_model(self.model)

    # metrics
    self.metrics = Metrics()

    # summary
    self.summary_writer = tf.summary.create_file_writer(logdir)

    # checkpoint
    self.ckpt_path = os.path.join(self.logdir, 'ckpts')

    try:
      init_lr = optimizer.lr.numpy()
      self.reduce_lr = ReduceLearningRate(
        init_lr, factor=factor, patience=patience, min_lr=min_lr)
    except Exception as e:
      print(e)

  def get_callbacks(self, logdir, profile_batch=0):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=logdir, histogram_freq=1, profile_batch=profile_batch)
    return [tensorboard_callback]

  def train(self, dataset, logging_steps, lr=None):
    self.logging_steps = logging_steps

    optimizer = self.optimizer
    loss_fn = self.loss_fn

    train_dataset = dataset.train_input_fn()
    valid_dataset = dataset.valid_input_fn()

    with self.summary_writer.as_default():
      # Train loop
      step = self._train(
        optimizer, loss_fn, train_dataset, valid_dataset, lr)

      # Validation for final
      self.valid(loss_fn, valid_dataset, step, True)

  def _train(
    self, optimizer, loss_fn, train_dataset,
    valid_dataset, lr=None):

    ckpt, manager, step = self.restore_checkpoints(optimizer)

    for (x_train, *y_train) in train_dataset:
      # manually update steps
      step += 1
      ckpt.step.assign(step)

      # training step
      y_true, y_pred = self.train_step(
        self.model, optimizer, loss_fn, x_train, y_train)

      # https://github.com/tensorflow/models/issues/7687
      data = optimizer._decayed_lr(var_dtype=tf.float32)
      tf.summary.scalar('Learning Rate', data=data, step=step)

      # train logging
      self.metrics.set_train_accuracy(y_true, y_pred)
      self.metrics.write_train(step)
      self.metrics.reset_train()

      if step % self.logging_steps == 0:
        start_time = time.time()
        # save checkpoints
        # 保存する処理はちょー重いのでまびく
        self.save_checkpoints(ckpt, manager, step)
        logger.info(f'save checkpoint: {time.time() - start_time:.4f} s')

        # validation for each steps
        start_time = time.time()
        self.valid(loss_fn, valid_dataset, step)
        logger.info(f'total validation: {time.time() - start_time:.4f} s')
    return step

  def valid(self, loss_fn, valid_dataset, step, output_image=False):
    # x_valids = None

    start_time = time.time()

    # validation
    for (x_valid, *y_valid) in valid_dataset:
      y_true, y_pred = self.valid_step(
        self.model, loss_fn, x_valid, y_valid)
      self.metrics.set_valid_accuracy(y_true, y_pred)

      # x_valids = np.array(x_valid.numpy()) if x_valids is None \
      #     else np.concatenate((x_valids, x_valid.numpy()))

      # for i, y in enumerate(y_true):
      #   y_trues[i] = np.array(y.numpy()) if y_trues[i] is None \
      #       else np.concatenate((y_trues[i], y.numpy()))
      # for i, y in enumerate(y_pred):
      #   y_preds[i] = np.array(y.numpy()) if y_preds[i] is None \
      #       else np.concatenate((y_preds[i], y.numpy()))

    # logger.info('validation size: {}'.format(len(x_valids)))
    logger.info(f'validation: {time.time() - start_time:.4f} s')

    start_time = time.time()
    self.metrics.write_valid(step)
    val_accuracy = self.metrics.get_valid_accuracy()
    self.metrics.reset_valid()
    logger.info(f'save metrics: {time.time() - start_time:.4f} s')

    # for c, v in zip([*self.labels, 'total'], recalls):
    #   logger.info('Recall(valid): {}={:.4f}'.format(c, v))
    # self.save_best_score(recalls[-1])

    epoch = step // self.logging_steps

    if hasattr(self, 'reduce_lr'):
      new_lr = self.reduce_lr.new_lr(epoch, val_accuracy[1].numpy())
      self.optimizer.lr.assign(new_lr)

  def save(self):
    dirname = datetime.datetime.now().strftime('%s')
    path = os.path.join(self.logdir, 'models', dirname)
    os.makedirs(path, exist_ok=True)

    # TODO: どうも callback.on_epoch_end を読んでしまうと保存はできるが、
    # load_modelできないのであとで考える
    self.model.save(path, save_format='tf', include_optimizer=False)

  def save_best_score(self, score):
    if score > self.best_threshold:
      dirname = datetime.datetime.now().strftime('%s')
      dirname = '{}-{:.4f}'.format(dirname, score)
      path = os.path.join(self.logdir, 'models', dirname)
      os.makedirs(path, exist_ok=True)
      tf.saved_model.save(self.model, path)
      self.best_threshold = score

  def restore_checkpoints(self, optimizer):
    step = 0
    ckpt = tf.train.Checkpoint(
      step=tf.Variable(step, tf.int64), optimizer=optimizer, net=self.model)
    manager = tf.train.CheckpointManager(ckpt, self.ckpt_path, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
      step = ckpt.step.numpy()
      logger.info(
        'Restore from ckpt: {}, step={}'.format(
          manager.latest_checkpoint, step))
    else:
      logger.info('Initialize from scratch')
    return ckpt, manager, step

  def save_checkpoints(self, ckpt, manager, step):
    ckpt.step.assign(step)
    save_path = manager.save()
    logger.info('Save checkpoint for step {}: {}'.format(
      int(ckpt.step), save_path))

  @tf.function
  def train_step(self, model, optimizer, loss_fn, x_train, y_train):
    with tf.GradientTape() as tape:
      y_pred = model(x_train, training=True)
      losses = loss_fn(y_train, y_pred)
      self.metrics.set_train_loss(losses)

    grads = tape.gradient(losses[::-1], model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return y_train, y_pred

  @tf.function
  def valid_step(self, model, loss_fn, x_valid, y_valid):
    y_pred = model(x_valid)
    losses = loss_fn(y_valid, y_pred)
    self.metrics.set_valid_loss(losses)
    return y_valid, y_pred
