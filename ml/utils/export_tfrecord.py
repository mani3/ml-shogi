import os
import glob
import pickle

import argparse
from tqdm import tqdm
import tensorflow as tf


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(feature, move_label, win, move_number, steps):
  feature = {
    'feature': _bytes_feature(feature.tobytes()),
    'height': _int64_feature(feature.shape[0]),
    'width': _int64_feature(feature.shape[1]),
    'channel': _int64_feature(feature.shape[2]),
    'move_label': _int64_feature(move_label),
    'win': _int64_feature(win),
    'move_number': _int64_feature(move_number),
    'steps': _int64_feature(steps)
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def main(args):
  input_pattern = os.path.join(args.input_dir, '*.pickle')
  input_paths = glob.glob(input_pattern)

  output_dir = os.path.dirname(args.output_path)
  os.makedirs(output_dir, exist_ok=True)

  with tf.io.TFRecordWriter(args.output_path) as writer:
    for path in tqdm(input_paths):
      with open(path, 'rb') as f:
        [x, y, win, move_number, steps] = pickle.load(f)
      example = serialize_example(x, y, win, move_number, steps)
      writer.write(example)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('input_dir', type=str)
  parser.add_argument('--output_path', type=str, default='/tmp/train.tfrecord')
  main(parser.parse_args())
