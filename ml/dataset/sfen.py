import copy
import absl

import numpy as np
import pandas as pd
import tensorflow as tf

import shogi
from ml.dataset import common

logger = absl.logging


class SFEN(object):

  def __init__(
    self, train_path, valid_path,
    epochs, batch_size=64, seed=42):

    self.batch_size = batch_size
    self.seed = seed
    self.epochs = epochs

    self.train_path = train_path
    self.valid_path = valid_path
    self.train_df = pd.read_csv(train_path)
    self.valid_df = pd.read_csv(valid_path)

    self.train_size = self.get_size(train_path)
    self.valid_size = self.get_size(valid_path)

    logger.info(
      f'train_count: {self.train_size}, valid_count: {self.valid_size}')

  def get_size(self, filename):
    with open(filename) as f:
      count = len(f.readlines())
      return count

  def make_inputs(self, sfen_list):
    inputs = None

    for sfen in sfen_list:
      sfen = sfen.numpy().decode('utf-8')
      x = make_inputs(sfen)
      x = x.transpose([1, 2, 0])
      x = np.expand_dims(x, axis=0)
      inputs = x if inputs is None else np.concatenate([inputs, x])
    return inputs

  def preprocess(self, sfens, wins, labels):
    x = tf.py_function(self.make_inputs, [sfens], [tf.float32])
    x = tf.reshape(x, [-1, 9, 9, 43])
    return x, labels, wins

  def train_input_fn(self):
    batch_size = self.batch_size
    df = self.train_df

    num_parallel = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices(
      (df['sfen'], df['win'], df['label']))

    dataset = (dataset.shuffle(10000, seed=self.seed)
               .batch(batch_size)
               .map(self.preprocess, num_parallel_calls=num_parallel)
               .repeat(self.epochs)
               .prefetch(tf.data.experimental.AUTOTUNE))
    return dataset

  def valid_input_fn(self):
    batch_size = self.batch_size
    df = self.valid_df

    num_parallel = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices(
      (df['sfen'], df['win'], df['label']))

    dataset = (dataset.repeat(1)
               .batch(batch_size)
               .map(self.preprocess, num_parallel_calls=num_parallel)
               .prefetch(tf.data.experimental.AUTOTUNE))
    return dataset


def make_inputs(sfen):
  board = shogi.Board(sfen=sfen)
  inputs = make_input_features(board)
  return inputs


def make_input_features(board):
  if board.turn == shogi.BLACK:
    piece_bb = copy.deepcopy(board.piece_bb)
    occupied = copy.deepcopy(
      (board.occupied[shogi.BLACK],
       board.occupied[shogi.WHITE])
    )
    pieces_in_hand = copy.deepcopy(
      (board.pieces_in_hand[shogi.BLACK],
       board.pieces_in_hand[shogi.WHITE])
    )
  else:
    piece_bb = [common.bb_rotate_180(bb) for bb in board.piece_bb]
    occupied = (
      common.bb_rotate_180(board.occupied[shogi.WHITE]),
      common.bb_rotate_180(board.occupied[shogi.BLACK])
    )
    pieces_in_hand = (
      board.pieces_in_hand[shogi.WHITE],
      board.pieces_in_hand[shogi.BLACK]
    )

  features = []
  for color in shogi.COLORS:
    # pieces on board
    for piece_type in shogi.PIECE_TYPES_WITH_NONE[1:]:
      bb = piece_bb[piece_type] & occupied[color]
      feature = np.zeros(9 * 9, dtype=np.uint8)
      for pos in shogi.SQUARES:
        if bb & shogi.BB_SQUARES[pos] > 0:
          feature[pos] = 1
      features.append(feature.reshape((9, 9)))

    # pieces in hand
    for piece_type in range(1, 8):
      if piece_type in pieces_in_hand[color]:
        n = pieces_in_hand[color][piece_type]
        f = np.full((9, 9), n, dtype=np.uint8)
      else:
        f = np.zeros((9, 9), dtype=np.uint8)
      features.append(f)

  move_num = min(board.move_number, np.iinfo(np.uint8).max)
  f = np.full((9, 9), move_num, dtype=np.uint8)
  features.append(f)

  return np.array(features)
