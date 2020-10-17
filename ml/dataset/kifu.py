import copy
import numpy as np

import shogi
import shogi.CSA

import ml.dataset.common as common
import ml.dataset.features as features


def read(kifu_list_file):
  feature_all = None
  move_all = []
  win_all = []

  with open(kifu_list_file, 'r') as f:
    for line in f.readlines():
      filepath = line.rstrip('\n')
      inputs, moves, wins = read_kifu(filepath)

      feature_all = inputs if feature_all is None else np.append(feature_all, inputs, axis=0)
      move_all.extend(moves)
      win_all.extend(wins)
  return (feature_all, np.array(move_all), np.array(win_all))


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
