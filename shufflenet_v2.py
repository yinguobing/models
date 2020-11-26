"""A human friendly ShuffleNet V2 implementation."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def shuffle_unit_v2(split=0.5, downsampling=False):
    """
    Build building blocks for ShuffleNet v2.

    Args:
        split: a float value in range (0, 1), indicating how many channels of
            the total input channels should be split into the identity path.
            This argument will be ignored when downsampling is True.
        downsampling: whether to downsample the input feature map by half.

    Returns:
        a callable ShuffleNet v2 block.
    """
    # Safety check.
    if not downsampling:
        assert split > 0 and split < 1, \
            "Split value should be in range (0, 1), got {}".format(split)

    strides = 2 if downsampling else 1

    def forward(inputs):
        # Input channels will be split in two paths. One for identity path and
        # the other for conv path. Assume the inputs are `CHANNEL LAST`.
        batch_size, height, width, num_input_channels = inputs.shape

        # Channel split is not applied while downsampling.
        if downsampling:
            # Do NOT split channels.
            x_idn = inputs
            x_conv = inputs

            # Number of filters equals to the input channels.
            filters = num_input_channels

            # Build downsampling layers for identity path.
            identity_path_layers = [keras.layers.DepthwiseConv2D(3, strides, 'same'),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.Conv2D(
                                    filters, 1, padding='same'),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.ReLU()]
        else:
            # Split channels.
            num_identity = tf.cast(
                tf.math.ceil(num_input_channels * split), tf.int32)

            # Avoid edge condition in which spliting channels failed.
            if num_identity == num_input_channels:
                num_identity -= 1

            x_idn = inputs[:, :, :, :num_identity]
            x_conv = inputs[:, :, :, num_identity:]

            # Number of filters equal to the conv path channels.
            filters = num_input_channels - num_identity

            # Build downsampling layers for identity path.
            identity_path_layers = []

        # Build layers for convolutional path.
        conv_path_layers = [keras.layers.Conv2D(filters, 1, padding='same'),
                            keras.layers.BatchNormalization(),
                            keras.layers.ReLU(),
                            keras.layers.DepthwiseConv2D(3, strides, 'same'),
                            keras.layers.BatchNormalization(),
                            keras.layers.Conv2D(filters, 1, padding='same'),
                            keras.layers.BatchNormalization(),
                            keras.layers.ReLU()]

        # Run forward on identity path.
        for layer in identity_path_layers:
            x_idn = layer(x_idn)

        # Run forward on convolutional path.
        for layer in conv_path_layers:
            x_conv = layer(x_conv)

        # Shuffle from the channel dimension.
        x = tf.concat([x_idn, x_conv], axis=-1)
        shuffled = shuffle(x)

        return shuffled

    return forward


def shuffle(x, groups=2):
    """Shuffle x from the channel dimension.

    Args:
        x: a tensor of format (batch_size, height, width, channels)
        groups: number of groups.

    Returns:
        a shuffled tensor.
    """
    batch_size, height, width, channels = x.shape
    channels_per_group = channels // groups

    x = tf.reshape(x, (batch_size, height, width, groups, channels_per_group))
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, (batch_size, height, width, -1))

    return x
