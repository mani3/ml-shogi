import os
import argparse
import random


def is_csa(filename):
  return os.path.splitext(filename)[1] == '.csa'


def get_all_csa_path(directory):
  filepaths = []
  for root, _, files in os.walk(directory):
    filepaths.extend([os.path.join(root, f) for f in files if is_csa(f)])
  return filepaths


def main(args):
  kifu_list = get_all_csa_path(args.dir)

  random.seed(args.seed)
  random.shuffle(kifu_list)

  train_len = int(len(kifu_list) * args.ratio)
  with open(args.filename + '_train.txt', 'w') as f:
    for path in kifu_list[:train_len]:
      f.write(path)
      f.write('\n')

  with open(args.filename + '_valid.txt', 'w') as f:
    for path in kifu_list[train_len:]:
      f.write(path)
      f.write('\n')

  print(f'total kifu size: {len(kifu_list)}')
  print(f'train kifu size: {train_len}')
  print(f'valid kifu size: {len(kifu_list) - train_len}')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('dir', type=str)
  parser.add_argument('filename', type=str)
  parser.add_argument('--ratio', type=float, default=0.9)
  parser.add_argument('--seed', type=int, default=42)
  main(parser.parse_args())
