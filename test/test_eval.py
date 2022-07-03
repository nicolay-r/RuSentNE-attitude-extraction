import unittest
from os.path import join, exists, dirname, realpath


class TestEval(unittest.TestCase):

    def test(self):

        current_dir = dirname(realpath(__file__))
        output_dir = join(current_dir, "_out")

        source_filename = "predict.tsv.gz"
        serialize_dir = "serialize-nn_3l"

        output_filename = join(output_dir, serialize_dir, source_filename)
        etalon_filename = join(output_dir, serialize_dir, "sample-etalon-0.tsv.gz")

        if not exists(output_filename):
            raise FileNotFoundError(output_filename)

        if not exists(etalon_filename):
            raise FileNotFoundError(etalon_filename)


if __name__ == '__main__':
    unittest.main()
