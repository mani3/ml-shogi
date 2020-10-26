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


def residual_block(
  inputs, filters, kernel_size, strides,
  activation_name='leaky_relu', data_format='channels_last'):

  shortcut = inputs

  initializer = tf.keras.initializers.TruncatedNormal(stddev=0.05)
  regularizer = tf.keras.regularizers.l2(0.01)
  padding = ('SAME' if strides == 1 else 'VALID')

  x = Conv2D(
    filters=filters, kernel_size=kernel_size,
    strides=strides, padding=padding,
    kernel_initializer=initializer,
    kernel_regularizer=regularizer,
    use_bias=False, data_format=data_format)(inputs)

  x = batch_norm(data_format)(x)
  x = activation(x, activation_name)

  x = Conv2D(
    filters=filters, kernel_size=kernel_size,
    strides=strides, padding=padding,
    kernel_initializer=initializer,
    kernel_regularizer=regularizer,
    use_bias=True, data_format=data_format)(x)

  x = batch_norm(data_format)(x)
  x = tf.keras.layers.add([x, shortcut])
  x = activation(x, activation_name)
  return x


def resnet(input_shape, classes, resnet_size=18,
           num_filters=256,
           kernel_size=3,
           strides=1,
           data_format='channels_last', **kwargs):

  activation_name = kwargs.get('activation_name', 'mish')

  inputs = tf.keras.layers.Input(input_shape)
  if data_format == 'channels_first':
    inputs = tf.transpose(a=inputs, perm=[0, 3, 1, 2])

  initializer = tf.keras.initializers.TruncatedNormal(stddev=0.05)
  regularizer = tf.keras.regularizers.l2(0.01)

  x = Conv2D(
    filters=num_filters, kernel_size=kernel_size,
    strides=strides, padding="SAME",
    kernel_initializer=initializer,
    kernel_regularizer=regularizer,
    use_bias=True, data_format=data_format)(inputs)
  x = batch_norm(data_format)(x)
  x = activation(x, activation_name)

  for _ in range(resnet_size):
    x = residual_block(
      x, filters=num_filters, kernel_size=kernel_size, strides=strides,
      activation_name=activation_name, data_format=data_format)

  policy = Conv2D(classes, kernel_size=1, padding='same')(x)
  policy = batch_norm(data_format)(policy)
  policy = activation(policy, activation_name)
  policy = Flatten(name='fc_policy')(policy)

  values = Conv2D(classes, kernel_size=1, padding='same')(x)
  values = batch_norm(data_format)(values)
  values = activation(values, activation_name)
  values = Flatten()(values)
  values = Dense(256, name='fc_final')(values)
  values = Dense(1, name='fc_value')(values)

  model = tf.keras.models.Model(
    inputs, [policy, values], name=f'resnet{resnet_size}')
  return model


ResNet5 = functools.partial(
  resnet,
  resnet_size=5,
  num_filters=256,
  kernel_size=3,
  strides=1)

ResNet10 = functools.partial(
  resnet,
  resnet_size=10,
  num_filters=256,
  kernel_size=3,
  strides=1)

ResNet20 = functools.partial(
  resnet,
  resnet_size=20,
  num_filters=256,
  kernel_size=3,
  strides=1)

ResNet40 = functools.partial(
  resnet,
  resnet_size=40,
  num_filters=256,
  kernel_size=3,
  strides=1)
