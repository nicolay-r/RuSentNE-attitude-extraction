import unittest
from os.path import dirname, realpath, join

from run_predict_nn import predict_nn
from run_train_nn import train_nn


class TestNetworkTraining(unittest.TestCase):

    def test_train_predict(self):

        current_dir = dirname(realpath(__file__))
        output_dir = join(current_dir, "_out")

        train_nn(output_dir=output_dir,
                 model_log_dir=join(current_dir, "_model"),
                 split_source=join(current_dir, "..", "data/split_fixed.txt"))

        predict_nn(extra_name_suffix="nn",
                   output_dir=output_dir)


if __name__ == '__main__':
    unittest.main()
