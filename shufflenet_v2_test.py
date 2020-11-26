import unittest

import tensorflow as tf

from shufflenet_v2 import shuffle_unit_v2


class TestShuffleNetV2(unittest.TestCase):

    def setUp(self):
        pass

    def test_unit_split(self):
        """Split value should be in range (0, 1)"""
        self.assertRaises(AssertionError, shuffle_unit_v2, 32, 0)
        self.assertRaises(AssertionError, shuffle_unit_v2, 32, 1)


if __name__ == "__main__":
    unittest.main()
