import functools
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten


class MobileNetV3:
  def __init__(self, input_shape, classes, alpha=1.0):
    self.input_shape = input_shape
    self.classes = classes
    self.alpha = alpha

  def relu6(self, x):
    return tf.nn.relu6(x)

  def hard_swish(self, x):
    return x * tf.nn.relu6(x + 3.0) / 6.0

  def conv_block(self, inputs, filters, kernel, strides,
                 activation_fn, data_format='channels_last'):
    axis = 1 if data_format == 'channels_first' else 3

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=axis)(x)
    return activation_fn(x)

  def squeeze(self, inputs):
    input_channels = int(inputs.shape[-1])

    x = GlobalAveragePooling2D()(inputs)
    x = Dense(input_channels, activation='relu')(x)
    x = Dense(input_channels, activation='hard_sigmoid')(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])
    return x

  def bottleneck(self, inputs, filters, kernel, e, s, squeeze,
                 activation_fn, data_format='channels_last'):
    axis = 1 if data_format == 'channels_first' else 3
    input_shape = inputs.shape

    tchannel = int(e)
    cchannel = int(self.alpha * filters)

    r = s == 1 and input_shape[3] == filters

    x = self.conv_block(inputs, tchannel, (1, 1), (1, 1), activation_fn)

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=axis)(x)
    x = activation_fn(x)

    if squeeze:
      x = self.squeeze(x)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=axis)(x)

    if r:
      x = Add()([x, inputs])
      x = activation_fn(x)

    return x

  def large(self, filters=256, e=256, **kwargs):
    inputs = tf.keras.layers.Input(shape=self.input_shape)

    x = self.conv_block(inputs, filters, (3, 3), (1, 1), Activation(self.relu6))

    x = self.bottleneck(x, filters, (3, 3), e=e, s=1, squeeze=False, activation_fn=Activation(self.relu6))
    x = self.bottleneck(x, filters, (3, 3), e=e, s=1, squeeze=False, activation_fn=Activation(self.relu6))
    x = self.bottleneck(x, filters, (3, 3), e=e, s=1, squeeze=False, activation_fn=Activation(self.relu6))
    x = self.bottleneck(x, filters, (3, 3), e=e, s=1, squeeze=True, activation_fn=Activation(self.relu6))
    x = self.bottleneck(x, filters, (3, 3), e=e, s=1, squeeze=True, activation_fn=Activation(self.relu6))
    x = self.bottleneck(x, filters, (3, 3), e=e, s=1, squeeze=True, activation_fn=Activation(self.relu6))
    x = self.bottleneck(x, filters, (3, 3), e=e, s=1, squeeze=False, activation_fn=Activation(self.hard_swish))
    x = self.bottleneck(x, filters, (3, 3), e=e, s=1, squeeze=False, activation_fn=Activation(self.hard_swish))
    x = self.bottleneck(x, filters, (3, 3), e=e, s=1, squeeze=False, activation_fn=Activation(self.hard_swish))
    x = self.bottleneck(x, filters, (3, 3), e=e, s=1, squeeze=False, activation_fn=Activation(self.hard_swish))
    x = self.bottleneck(x, filters, (3, 3), e=e, s=1, squeeze=True, activation_fn=Activation(self.hard_swish))
    x = self.bottleneck(x, filters, (3, 3), e=e, s=1, squeeze=True, activation_fn=Activation(self.hard_swish))
    x = self.bottleneck(x, filters, (3, 3), e=e, s=1, squeeze=True, activation_fn=Activation(self.hard_swish))
    x = self.bottleneck(x, filters, (3, 3), e=e, s=1, squeeze=True, activation_fn=Activation(self.hard_swish))
    x = self.bottleneck(x, filters, (3, 3), e=e, s=1, squeeze=True, activation_fn=Activation(self.hard_swish))

    policy = self.conv_block(x, self.classes, (3, 3), (1, 1), Activation(self.relu6))
    policy = Flatten(name='fc_policy')(policy)

    values = self.conv_block(x, self.classes, (1, 1), (1, 1), Activation(self.relu6))
    values = Flatten()(values)
    values = Dense(256, name='fc_final')(values)
    values = Dense(1, name='fc_value')(values)

    model = tf.keras.models.Model(
      inputs, [policy, values], name=f'mobilenetv3_large_{filters}')
    return model


def mobilenetv3_large(input_shape, classes, filters=256, alpha=1.0, **kwargs):
  mobilenet = MobileNetV3(input_shape, classes, alpha=alpha)
  return mobilenet.large(filters, **kwargs)


MobileNetV3Large = functools.partial(mobilenetv3_large)

MobileNetV3_192 = functools.partial(
  mobilenetv3_large,
  filters=192,
  e=512,
)

MobileNetV3_256 = functools.partial(
  mobilenetv3_large,
  filters=256,
  e=512,
)
