import unittest
from os.path import join, dirname, realpath

from arekit.contrib.utils.evaluation.results.three_class import ThreeClassEvalResult

from labels.scaler import PosNegNeuRelationsLabelScaler
from run_eval import text_opinion_monolith_collection_result_evaluator
from run_eval_doc import opinions_per_document_result_evaluation


class TestEval(unittest.TestCase):

    def test(self):
        """ We adopt AREkit API for this and relying on the following example:
            https://github.com/nicolay-r/AREkit/blob/master/tests/contrib/utils/test_eval.py
        """

        current_dir = dirname(realpath(__file__))
        output_dir = join(current_dir, "_out")

        source_filename = "predict-cnn.tsv.gz"
        serialize_dir = "serialize-nn_3l"

        predict_filename = join(output_dir, serialize_dir, source_filename)
        test_filename = join(output_dir, serialize_dir, "sample-test-0.tsv.gz")
        etalon_filename = join(output_dir, serialize_dir, "sample-etalon-0.tsv.gz")

        result = text_opinion_monolith_collection_result_evaluator(
            test_predict_filename=predict_filename,
            etalon_samples_filepath=etalon_filename,
            test_samples_filepath=test_filename,
            label_scaler=PosNegNeuRelationsLabelScaler())

        print(result.TotalResult)
        print(result.get_result_by_metric(ThreeClassEvalResult.C_F1))
        print(result.get_result_by_metric(ThreeClassEvalResult.C_F1_POS))
        print(result.get_result_by_metric(ThreeClassEvalResult.C_F1_NEG))

    def test_opinion_eval(self):

        current_dir = dirname(realpath(__file__))
        output_dir = join(current_dir, "_out")

        source_filename = "predict-cnn.tsv.gz"
        serialize_dir = "serialize-nn_3l"

        predict_filename = join(output_dir, serialize_dir, source_filename)
        test_filename = join(output_dir, serialize_dir, "sample-test-0.tsv.gz")
        etalon_filename = join(output_dir, serialize_dir, "sample-etalon-0.tsv.gz")

        result = opinions_per_document_result_evaluation(
            test_predict_filename=predict_filename,
            etalon_samples_filepath=etalon_filename,
            test_samples_filepath=test_filename,
            label_scaler=PosNegNeuRelationsLabelScaler())

        print(result.TotalResult)
        print(result.get_result_by_metric(ThreeClassEvalResult.C_F1))
        print(result.get_result_by_metric(ThreeClassEvalResult.C_F1_POS))
        print(result.get_result_by_metric(ThreeClassEvalResult.C_F1_NEG))


if __name__ == '__main__':
    unittest.main()
