import unittest

from run_train_nn import train_nn


class TestNetworkTraining(unittest.TestCase):

    def test(self):
        train_nn(extra_name_suffix="test-nn")


if __name__ == '__main__':
    unittest.main()
