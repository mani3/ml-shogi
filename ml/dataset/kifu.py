import os
import copy
import pickle
import hashlib

import numpy as np
from tqdm import tqdm

import shogi
import shogi.CSA

import ml.dataset.common as common
import ml.dataset.features as features


def read(kifu_list_file):
  feature_all = []
  move_all = []
  win_all = []

  with open(kifu_list_file, 'r') as f:
    for line in tqdm(f.readlines()):
      filepath = line.rstrip('\n')
      inputs, moves, wins = read_kifu(filepath)

      feature_all.append(inputs)
      move_all.extend(moves)
      win_all.extend(wins)

  feature_all = np.concatenate(feature_all, axis=0)
  move_all = np.asarray(move_all)
  win_all = np.asarray(win_all)
  return (feature_all, win_all, win_all)


def read_kifu(filepath):
  kifu = shogi.CSA.Parser.parse_file(filepath)[0]
  win_color = shogi.BLACK if kifu['win'] == 'b' else shogi.WHITE
  board = shogi.Board()

  move_labels = []
  wins = []
  feature_np = None

  for move in kifu['moves']:
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
        (common.bb_rotate_180(board.occupied[shogi.WHITE]),
         common.bb_rotate_180(board.occupied[shogi.BLACK]))
      )
      pieces_in_hand = copy.deepcopy(
        (board.pieces_in_hand[shogi.WHITE],
         board.pieces_in_hand[shogi.BLACK])
      )

    move_label = features.make_output_label(
      shogi.Move.from_usi(move), board.turn)

    win = 1 if win_color == board.turn else 0

    wins.append(win)
    move_labels.append(move_label)
    feature = features.make_input_features(piece_bb, occupied, pieces_in_hand)

    if feature_np is None:
      feature_np = np.expand_dims(feature, axis=0)
    else:
      feature_np = np.append(
        feature_np, np.expand_dims(feature, axis=0), axis=0)

    board.push_usi(move)
  return (feature_np, move_labels, wins)


class Phase(object):

  def __init__(self, feature, label, win, steps, move_number):
    self.feature = self.convert_np(feature)
    self.label = label
    self.win = win
    self.steps = steps
    self.move_number = move_number

  def convert_np(self, x):
    if not x.flags['C_CONTIGUOUS']:
      x = np.ascontiguousarray(x)
    return x

  def data(self):
    return [
      self.feature,
      self.label,
      self.win,
      self.move_number,
      self.steps,
    ]

  def save(self, export_dir):
    key = np.array_str(self.feature) + str(self.label) + str(self.win)
    hash = hashlib.sha256(key.encode('utf8')).hexdigest()
    output_path = os.path.join(export_dir, f'{hash}.pickle')

    os.makedirs(export_dir, exist_ok=True)
    if os.path.exists(output_path):
      # print(f'Already exist: {output_path}, {self.move_number}, {self.steps}')
      pass
    else:
      with open(output_path, 'wb') as f:
        pickle.dump(self.data(), f, pickle.HIGHEST_PROTOCOL)


def export_data(filepath, export_dir):
  board = shogi.Board()

  kifu = shogi.CSA.Parser.parse_file(filepath)[0]
  win_color = shogi.BLACK if kifu['win'] == 'b' else shogi.WHITE

  count = len(kifu['moves'])
  sfen = kifu['sfen']

  for move in kifu['moves']:
    win = 1 if win_color == board.turn else 0
    move_number = board.move_number
    feature = features.make_input_features_from_board(board)
    move_label = features.make_output_label(shogi.Move.from_usi(move), board.turn)  # noqa: E501

    phase = Phase(feature, move_label, win, count, move_number)
    phase.save(export_dir)
    board.push_usi(move)
  # print(f'Completion: steps={count}, sfen={sfen}')
