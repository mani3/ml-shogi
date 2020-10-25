import functools
import tensorflow as tf

from tensorflow.keras.layers import (
  Input,
  Conv2D,
  BatchNormalization,
  LeakyReLU,
  Flatten,
  Dense,
)

from ml.dataset.common import MOVE_DIRECTION_LABEL_NUM


def simple(
  input_shape=(9, 9, 104),
  classes=MOVE_DIRECTION_LABEL_NUM,
  filters=192,
  kernel_size=3,
  bn_axis=3, **kargs):

  inputs = Input(input_shape)

  x = Conv2D(filters, kernel_size=kernel_size, padding='same')(inputs)
  x = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
  x = BatchNormalization(axis=bn_axis)(x)
  x = LeakyReLU(alpha=0.1)(x)

  x = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
  x = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
  x = BatchNormalization(axis=bn_axis)(x)
  x = LeakyReLU(alpha=0.1)(x)

  x = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
  x = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
  x = BatchNormalization(axis=bn_axis)(x)
  x = LeakyReLU(alpha=0.1)(x)

  x = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
  x = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
  x = BatchNormalization(axis=bn_axis)(x)
  x = LeakyReLU(alpha=0.1)(x)

  x = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
  x = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
  x = BatchNormalization(axis=bn_axis)(x)
  x = LeakyReLU(alpha=0.1)(x)

  x = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
  x = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
  x = BatchNormalization(axis=bn_axis)(x)
  x = LeakyReLU(alpha=0.1)(x)

  policy = Conv2D(classes, kernel_size=1, padding='same')(x)
  policy = Flatten()(policy)

  values = Conv2D(classes, kernel_size=1, padding='same')(x)
  values = Flatten()(values)
  values = Dense(256)(values)
  values = Dense(1)(values)

  model = tf.keras.models.Model(inputs, [policy, values], name='simple_cnn')
  return model


simple192 = functools.partial(simple, filters=192)
simple256 = functools.partial(simple, filters=256)
