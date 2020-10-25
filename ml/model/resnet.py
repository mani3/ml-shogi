import functools
import tensorflow as tf
# import tensorflow_addons as tfa

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization

from ml.model.mish import mish

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm(data_format='channels_last'):
  axis = 1 if data_format == 'channels_first' else 3
  momentum = _BATCH_NORM_DECAY
  epsilon = _BATCH_NORM_EPSILON
  return BatchNormalization(axis, momentum, epsilon, fused=True)


def activation(x, name: str, **kwargs):
  if name == 'leaky_relu':
    return tf.nn.leaky_relu(x, alpha=0.1)
  elif name == 'mish':
    return mish(x)
  elif name == 'relu':
    return tf.nn.relu(x)
  else:
    raise ValueError(f'Not found activation function name: {name}')


def fixed_padding(inputs, kernel_size, data_format='channels_last'):
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == 'channels_first':
    padded_inputs = tf.pad(
      tensor=inputs, paddings=[
        [0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(
      tensor=inputs, paddings=[
        [0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  padding = ('SAME' if strides == 1 else 'VALID')
  initializer = tf.keras.initializers.VarianceScaling()

  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)
  x = Conv2D(
    filters=filters, kernel_size=kernel_size,
    strides=strides, padding=padding,
    use_bias=False, kernel_initializer=initializer,
    data_format=data_format)(inputs)
  return x


def building_block_v2(
  inputs, filters, strides, data_format, projection_shortcut,
  activation_name='leaky_relu'):
  shortcut = inputs
  x = batch_norm(data_format)(inputs)
  x = activation(x, activation_name)

  if projection_shortcut:
    shortcut = conv2d_fixed_padding(x, filters, 1, strides, data_format)

  x = conv2d_fixed_padding(x, filters, 1, strides, data_format)
  x = batch_norm(data_format)(x)
  x = activation(x, activation_name)
  x = batch_norm(data_format)(x)
  return x + shortcut


def bottleneck_block_v2(
  inputs, filters, strides, data_format, projection_shortcut,
  activation_name='leaky_relu'):
  shortcut = inputs
  x = batch_norm(data_format)(inputs)
  x = activation(x, activation_name)

  if projection_shortcut:
    shortcut = conv2d_fixed_padding(x, filters, 1, strides, data_format)

  x = conv2d_fixed_padding(x, filters, 1, 1, data_format)

  x = batch_norm(data_format)(x)
  x = activation(x, activation_name)
  x = conv2d_fixed_padding(x, filters, 3, strides, data_format)

  x = batch_norm(data_format)(x)
  x = activation(x, activation_name)
  x = conv2d_fixed_padding(x, filters, 1, 1, data_format)
  return x + shortcut


def block_layer(
  inputs, filters, bottleneck, blocks, strides, name, data_format,
  activation_name='leaky_relu'):
  if bottleneck:
    block_fn = bottleneck_block_v2
  else:
    block_fn = building_block_v2

  x = block_fn(inputs, filters, strides, data_format, True, activation_name)

  for _ in range(1, blocks):
    x = block_fn(
      x, filters, 1, data_format, False, activation_name)
  return tf.identity(x, name)


def resnet(input_shape, classes, resnet_size=18,
           num_filters=256,
           kernel_size=3,
           conv_stride=1,
           block_sizes=[2, 2, 2, 2],
           block_strides=[1, 1, 1, 1],
           data_format='channels_last', **kwargs):

  bottleneck = False if resnet_size < 50 else True
  activation_name = kwargs.get('activation_name', 'leaky_relu')

  inputs = tf.keras.layers.Input(input_shape)
  if data_format == 'channels_first':
    inputs = tf.transpose(a=inputs, perm=[0, 3, 1, 2])

  x = conv2d_fixed_padding(
    inputs, num_filters, kernel_size, conv_stride, data_format)

  for i, num_blocks in enumerate(block_sizes):
    filters = num_filters  # * (2**i)
    x = block_layer(
      x, filters=filters, bottleneck=bottleneck, blocks=num_blocks,
      strides=block_strides[i], name='block_layer{}'.format(i + 1),
      data_format=data_format, activation_name=activation_name)

  x = batch_norm(data_format)(x)

  policy = Conv2D(classes, kernel_size=1, padding='same')(x)
  policy = Flatten(name='fc_policy')(policy)

  values = Conv2D(classes, kernel_size=1, padding='same')(x)
  values = Flatten()(values)
  values = Dense(256, name='fc_final')(values)
  values = Dense(1, name='fc_value')(values)

  model = tf.keras.models.Model(
    inputs, [policy, values], name=f'resnet{resnet_size}')
  return model


ResNet18 = functools.partial(
  resnet,
  resnet_size=18,
  num_filters=256,
  kernel_size=3,
  conv_stride=1,
  block_sizes=[2, 2, 2, 2])

ResNet34 = functools.partial(
  resnet,
  resnet_size=34,
  num_filters=256,
  kernel_size=3,
  conv_stride=1,
  block_sizes=[3, 4, 6, 3])

ResNet30 = functools.partial(
  resnet,
  resnet_size=50,
  num_filters=256,
  kernel_size=3,
  conv_stride=1,
  block_sizes=[2, 2, 2, 2])

ResNet50 = functools.partial(
  resnet,
  resnet_size=50,
  num_filters=256,
  kernel_size=3,
  conv_stride=1,
  block_sizes=[3, 4, 6, 3])

ResNet101 = functools.partial(
  resnet,
  resnet_size=101,
  num_filters=256,
  kernel_size=3,
  conv_stride=1,
  block_sizes=[3, 4, 23, 3])


if __name__ == "__main__":
  net = ResNet34(input_shape=[128, 128, 1], num_classes=10)
  print(net.summary())
  net = ResNet50(input_shape=[128, 128, 1], num_classes=10, pooling_type='avg')
  print(net.summary())
  net = ResNet50(input_shape=[128, 128, 1], num_classes=10)
  print(net.summary())
  net = ResNet101(input_shape=[128, 128, 1], num_classes=10)
  print(net.summary())
