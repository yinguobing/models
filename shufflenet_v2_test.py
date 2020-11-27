import unittest

import tensorflow as tf

from shufflenet_v2 import shuffle_unit_v2


class TestShuffleNetV2(unittest.TestCase):

    def setUp(self):
        pass

    def test_unit_split(self):
        """Split value should be in range (0, 1)"""
        self.assertRaises(AssertionError, shuffle_unit_v2, 0, False)
        self.assertRaises(AssertionError, shuffle_unit_v2, 1, False)

    def test_downsampling(self):
        """Output feature map size should be downsampled by factor of 2."""
        x = tf.random.normal((8, 64, 48, 256))

        y = shuffle_unit_v2(0.5, False)(x)
        self.assertTrue(x.shape == y.shape)

        y = shuffle_unit_v2(0.5, True)(x)
        bs, h, w, c = x.shape
        self.assertListEqual([bs, h/2, w/2, c*2], y.shape.as_list())

    def test_filters(self):
        """Number of channels could be set by user combined with downsampling."""
        x = tf.random.normal((8, 64, 48, 256))

        y = shuffle_unit_v2(0.5, True, 32)(x)
        bs, h, w, _ = x.shape
        self.assertListEqual([bs, h/2, w/2, 32], y.shape.as_list())

        y = shuffle_unit_v2(0.5, False, 32)(x)
        self.assertEqual(x.shape, y.shape)


if __name__ == "__main__":
    unittest.main()
