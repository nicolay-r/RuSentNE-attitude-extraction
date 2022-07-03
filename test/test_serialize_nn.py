import unittest
from os.path import dirname, realpath, join

from run_serialize_nn import serialize_nn


class TestSerializeNeuralNetwork(unittest.TestCase):

    def test(self):

        current_dir = dirname(realpath(__file__))
        output_dir = join(current_dir, "_out")

        serialize_nn(limit=1, output_dir=output_dir)


if __name__ == '__main__':
    unittest.main()
