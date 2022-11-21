import unittest
from os.path import dirname, realpath, join

from framework.arenets.predict import predict_nn
from framework.arenets.train import train_nn


class TestNetworkTraining(unittest.TestCase):

    def test_train_predict(self):

        current_dir = dirname(realpath(__file__))
        output_nn_dir = join(current_dir, "_out", "serialize-nn")

        train_nn(output_dir=output_nn_dir,
                 model_log_dir=join(current_dir, "_model"),
                 split_filepath=join(current_dir, "../..", "data/split_fixed.txt"))

        predict_nn(output_dir=output_nn_dir, embedding_dir=output_nn_dir, samples_dir=output_nn_dir)


if __name__ == '__main__':
    unittest.main()
