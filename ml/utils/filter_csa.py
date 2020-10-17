import argparse
import os
import re
import statistics


def find_all_files(directory):
  for root, _, files in os.walk(directory):
    for f in files:
      yield os.path.join(root, f)


def main(args):
  rate_matcher = re.compile(r"^'(black|white)_rate:.*:(.*)$")

  kifu_count = 0
  rates = []

  for filepath in find_all_files(args.dir):
    rate = {}
    move_len = 0
    toryo = False

    with open(filepath, 'r', encoding='utf-8') as f:
      for line in f:
        line = line.strip()

        m = rate_matcher.match(line)
        if m:
          rate[m.group(1)] = float(m.group(2))

        if line[:1] == '+' or line[:1] == '-':
          move_len += 1

        if line == '%TORYO':
          toryo = True

    if (not toryo or move_len <= 50 or len(rate) < 2 or min(rate.values()) < 2500):
      os.remove(filepath)
    else:
      kifu_count += 1
      rates.extend([_ for _ in rate.values()])

  print(f'kifu count:  {kifu_count}')
  print(f'rate mean:   {statistics.mean(rates)}')
  print(f'rate median: {statistics.median(rates)}')
  print(f'rate max:    {max(rates)}')
  print(f'rate min:    {min(rates)}')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('dir', type=str)
  main(parser.parse_args())
