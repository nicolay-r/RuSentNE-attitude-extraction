import unittest

from run_serialize_nn import serialize_nn


class TestSerializeNeuralNetwork(unittest.TestCase):

    def test(self):
        serialize_nn(limit=1, suffix="nn-test")


if __name__ == '__main__':
    unittest.main()
