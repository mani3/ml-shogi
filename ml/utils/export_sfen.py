import argparse
import pandas as pd
from tqdm import tqdm

import shogi
import shogi.CSA

from ml.dataset import features


def parse_csa(filepath):
  board = shogi.Board()
  kifu = shogi.CSA.Parser.parse_file(filepath)[0]
  win_color = shogi.BLACK if kifu['win'] == 'b' else shogi.WHITE

  wins, sfens, moves, labels = [], [], [], []

  for move in kifu['moves']:
    win = 1 if win_color == board.turn else 0
    move_label = features.make_output_label(
      shogi.Move.from_usi(move), board.turn)

    wins.append(win)
    sfens.append(board.sfen())
    moves.append(move)
    labels.append(move_label)

    board.push_usi(move)
  return wins, sfens, moves, labels


def main(args):
  kifu_list_path = args.list_path
  export_path = args.export_path

  data = {
    'win': [], 'sfen': [], 'move': [], 'label': []
  }

  with open(kifu_list_path, 'r') as f:
    for line in tqdm(f.readlines()):
      filepath = line.rstrip('\n')
      res = parse_csa(filepath)
      data['win'].extend(res[0])
      data['sfen'].extend(res[1])
      data['move'].extend(res[2])
      data['label'].extend(res[3])

  df = pd.DataFrame(data)
  df = df[~df.duplicated()]
  df.to_csv(export_path, index=False)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('list_path', type=str)
  parser.add_argument('--export_path', type=str, default='/tmp/train.csv')
  main(parser.parse_args())
