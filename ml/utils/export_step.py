import argparse
from tqdm import tqdm

from ml.dataset import kifu


def main(args):
  kifu_list_path = args.list_path
  export_dir = args.export_dir

  with open(kifu_list_path, 'r') as f:
    for line in tqdm(f.readlines()):
      filepath = line.rstrip('\n')
      kifu.export_data(filepath, export_dir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('list_path', type=str)
  parser.add_argument('--export_dir', type=str, default='/tmp/train')
  main(parser.parse_args())
