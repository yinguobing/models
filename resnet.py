
"""A human friendly ResNet implementation."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers


def residual_block(inputs, filters=64, kernel_size=(3, 3), strides=(1, 1),
                   padding='same', activation='relu'):
    """Building block for shallow ResNet."""
    # Note down sampling is performed by the first conv layer. Batch
    # normalization (BN) adopted right after each convolution and before
    # activation.
    x = layers.Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding,
                      activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    # Then the second conv layer without down sampling.
    x = layers.Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=(1, 1),
                      padding=padding,
                      activation=None)(x)
    x = layers.BatchNormalization()(x)

    # Time for the famous shortcut connection. Down sample the input if the
    # feature maps are down sampled.
    if strides != (1, 1):
        inputs = layers.Conv2D(filters=filters,
                               kernel_size=(1, 1),
                               strides=strides,
                               padding=padding,
                               activation=None)(inputs)
        inputs = layers.BatchNormalization()(inputs)

    x = layers.Add()([x, inputs])

    # Finally, output of the block.
    x = layers.Activation(activation)(x)

    return x


def bottleneck_block(inputs, filters=64, expantion=1, kernel_size=(3, 3),
                     strides=(2, 2), padding='same', activation='relu'):
    """Building block for deeper ResNet."""
    # Bottleneck block could be expanded. Get the expanded size.
    filters *= expantion

    # Note down sampling is performed by the first conv layer. Batch
    # normalization (BN) adopted right after each convolution and before
    # activation.
    x = layers.Conv2D(filters=filters,
                      kernel_size=(1, 1),
                      strides=strides,
                      padding=padding,
                      activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    # Then the second conv layer without down sampling.
    x = layers.Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=(1, 1),
                      padding=padding,
                      activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    # Then the third conv layer, also without down sampling. This layer also
    # has 4 times more filters.
    x = layers.Conv2D(filters=filters * 4,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      padding=padding,
                      activation=None)(x)
    x = layers.BatchNormalization()(x)

    # Time for the famous shortcut connection. Down sample the input if the
    # feature maps are down sampled.
    if strides != (1, 1):
        inputs = layers.Conv2D(filters=filters * 4,
                               kernel_size=(1, 1),
                               strides=strides,
                               padding=padding,
                               activation=None)(inputs)
        inputs = layers.BatchNormalization()(inputs)

    x = layers.Add()([x, inputs])

    # Finally, output of the block.
    x = layers.Activation(activation)(x)

    return x


def make_resnet18(input_shape, output_size=1000):
    """Construct a ResNet18 model"""
    inputs = keras.Input(shape=input_shape, name="input_image_tensor")

    # Layer: conv1
    x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                      padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                            padding='same')(x)

    # Layer: conv2
    x = residual_block(x, filters=64, kernel_size=(3, 3), strides=(1, 1))
    x = residual_block(x, filters=64, kernel_size=(3, 3), strides=(1, 1))

    # Layer: conv3
    x = residual_block(x, filters=128, kernel_size=(3, 3), strides=(2, 2))
    x = residual_block(x, filters=128, kernel_size=(3, 3), strides=(1, 1))

    # Layer: conv4
    x = residual_block(x, filters=256, kernel_size=(3, 3), strides=(2, 2))
    x = residual_block(x, filters=256, kernel_size=(3, 3), strides=(1, 1))

    # Layer: conv5
    x = residual_block(x, filters=512, kernel_size=(3, 3), strides=(2, 2))
    x = residual_block(x, filters=512, kernel_size=(3, 3), strides=(1, 1))

    # Last layer.
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(output_size)(x)

    # Assemble the model.
    model = Model(inputs, outputs, name='resnet18')

    return model


class ResidualBlock(layers.Layer):
    def __init__(self, filters=64, downsample=False, activation='relu', **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.activation_fun = activation
        self.downsample = downsample

        # First conv layer with/without down sampling.
        strides = (2, 2) if downsample else (1, 1)
        self.conv2d_1 = layers.Conv2D(filters=filters,
                                      kernel_size=(3, 3),
                                      strides=strides,
                                      padding='same',
                                      activation=None)

        # Second Conv layer without down sampling.
        self.conv2d_2 = layers.Conv2D(filters=filters,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      activation=None)

        # Batch normalization layer.
        self.batch_norm_1 = layers.BatchNormalization()
        self.batch_norm_2 = layers.BatchNormalization()
        self.batch_norm_3 = layers.BatchNormalization()

        # Activation layer.
        self.Activation = layers.Activation

        # Shortcut connection.
        self.shortcut = layers.Add()

        # In case the inputs are down sampled.
        if downsample:
            self.downsample_inputs = layers.Conv2D(filters=filters,
                                                   kernel_size=(1, 1),
                                                   strides=strides,
                                                   padding='same',
                                                   activation=None)

    def call(self, inputs):
        # First conv.
        x = self.conv2d_1(inputs)
        x = self.batch_norm_1(x)
        x = self.Activation(self.activation_fun)(x)

        # Second conv.
        x = self.conv2d_2(x)
        x = self.batch_norm_2(x)

        # Shortcut.
        if self.downsample:
            inputs = self.downsample_inputs(inputs)
            inputs = self.batch_norm_3(inputs)
        x = self.shortcut([x, inputs])

        # Output.
        x = self.Activation(self.activation_fun)(x)

        return x


class ResidualBlocks(layers.Layer):
    """A bunch of Residual Blocks. Down sampling is only performed by the first 
    block if required."""

    def __init__(self, num_blocks=2, filters=64, downsample=False,
                 activation='relu', **kwargs):
        super(ResidualBlocks, self).__init__(**kwargs)
        self.block_1 = ResidualBlock(filters, downsample, activation)
        self.blocks = [ResidualBlock(filters, False, activation)
                       for _ in range(num_blocks - 1)]

    def call(self, inputs):
        x = self.block_1(inputs)
        for block in self.blocks:
            x = block(x)

        return x


class BottleneckBlock(layers.Layer):

    def __init__(self, filters=64, downsample=False, activation='relu', **kwargs):
        super(BottleneckBlock, self).__init__(**kwargs)

        self.activation_fun = activation
        self.downsample = downsample

        # First conv layer with/without down sampling.
        strides = (2, 2) if downsample else (1, 1)
        self.conv2d_1 = layers.Conv2D(filters=filters,
                                      kernel_size=(1, 1),
                                      strides=strides,
                                      padding='same',
                                      activation=None)

        # Second Conv layer without down sampling.
        self.conv2d_2 = layers.Conv2D(filters=filters,
                                      kernel_size=(3, 3),
                                      strides=(1, 1),
                                      padding='same',
                                      activation=None)

        # Third Conv layer without down sampling.
        self.conv2d_3 = layers.Conv2D(filters=filters*4,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding='same',
                                      activation=None)

        # Batch normalization layer.
        self.batch_norm_1 = layers.BatchNormalization()
        self.batch_norm_2 = layers.BatchNormalization()
        self.batch_norm_3 = layers.BatchNormalization()
        self.batch_norm_4 = layers.BatchNormalization()

        # Activation layer.
        self.Activation = layers.Activation

        # Shortcut connection.
        self.shortcut = layers.Add()

        # In case the inputs are down sampled.
        if downsample:
            self.match_inputs = layers.Conv2D(filters=filters * 4,
                                              kernel_size=(1, 1),
                                              strides=strides,
                                              padding='same',
                                              activation=None)

    def call(self, inputs):
        # First conv.
        x = self.conv2d_1(inputs)
        x = self.batch_norm_1(x)
        x = self.Activation(self.activation_fun)(x)

        # Second conv.
        x = self.conv2d_2(x)
        x = self.batch_norm_2(x)
        x = self.Activation(self.activation_fun)(x)

        # Third conv.
        x = self.conv2d_3(x)
        x = self.batch_norm_3(x)
        x = self.Activation(self.activation_fun)(x)

        # Shortcut.
        if self.downsample:
            inputs = self.match_inputs(inputs)
            inputs = self.batch_norm_4(inputs)
        x = self.shortcut([x, inputs])

        # Output.
        x = self.Activation(self.activation_fun)(x)

        return x


class ResNet18(Model):
    def __init__(self, output_size=1000, **kwargs):
        super(ResNet18, self).__init__(**kwargs)

        # Conv1
        self.conv1 = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                                   padding='same')
        self.maxpool2d = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                             padding='same')

        # Conv2
        self.residual_blocks_1 = ResidualBlocks(2, filters=64, downsample=False,
                                                name="conv2")

        # Conv3
        self.residual_blocks_2 = ResidualBlocks(2, filters=128, downsample=True,
                                                name="conv3")

        # Conv4
        self.residual_blocks_3 = ResidualBlocks(2, filters=256, downsample=True,
                                                name="conv4")

        # Conv5
        self.residual_blocks_4 = ResidualBlocks(2, filters=512, downsample=True,
                                                name="conv5")

        # Output
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(output_size)

    def call(self, inputs):
        # Conv1
        x = self.conv1(inputs)
        x = self.maxpool2d(x)

        # Conv2
        x = self.residual_blocks_1(x)

        # Conv3
        x = self.residual_blocks_2(x)

        # Conv4
        x = self.residual_blocks_3(x)

        # Conv5
        x = self.residual_blocks_4(x)

        # Output
        x = self.global_avg_pool(x)
        x = self.fc(x)

        return x

    def prepare_summary(self, input_shape):
        """Prepare the subclassed model for summary.Check out TensorFlow issue 
        29132 for the original code.

        Args:
            input_shape: A shape tuple (integers), not including the batch size.
        """
        self.build((1,) + input_shape)
        self.call(keras.Input(input_shape))
