"""A human friendly ShuffleNet V2 implementation."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def shuffle_unit_v2(filters, split=0.5, downsampling=False):
    """
    Build building blocks for ShuffleNet v2.

    Args:
        filters: number of filters(channels)
        split: a float value in range (0, 1), indicating how many channels of 
            the total input channels should be split into the identity path.
        downsampling: whether to downsample the input feature map by half.

    Returns:
        a callable ShuffleNet v2 block.
    """
    assert split > 0 and split < 1, "Split value should be in range (0, 1), got {}".format(
        split)
