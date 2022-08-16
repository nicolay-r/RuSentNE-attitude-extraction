import unittest
from os.path import join, dirname, realpath
from arekit.contrib.utils.evaluation.results.two_class_prf import TwoClassEvalPrecRecallF1Result

from evaluation.eval_document_level import opinions_per_document_two_class_result_evaluation
from evaluation.eval_instance_level import text_opinion_per_collection_result_evaluator
from labels.scaler import PosNegNeuRelationsLabelScaler


class TestEval(unittest.TestCase):

    @staticmethod
    def __print_result(result):
        print(result.TotalResult)
        print(result.get_result_by_metric(TwoClassEvalPrecRecallF1Result.C_F1))
        print(result.get_result_by_metric(TwoClassEvalPrecRecallF1Result.C_F1_POS))
        print(result.get_result_by_metric(TwoClassEvalPrecRecallF1Result.C_F1_NEG))

    @staticmethod
    def __create_data():
        current_dir = dirname(realpath(__file__))
        output_dir = join(current_dir, "_out")

        source_filename = "predict-cnn.tsv.gz"
        serialize_dir = "serialize-nn_3l"

        predict_filename = join(output_dir, serialize_dir, source_filename)
        test_filename = join(output_dir, serialize_dir, "sample-test-0.tsv.gz")
        etalon_filename = join(output_dir, serialize_dir, "sample-etalon-0.tsv.gz")

        return predict_filename, test_filename, etalon_filename

    def test_text_opinions_eval(self):
        """ We adopt AREkit API for this and relying on the following example:
            https://github.com/nicolay-r/AREkit/blob/master/tests/contrib/utils/test_eval.py
        """

        predict_filename, test_filename, etalon_filename = self.__create_data()

        result = text_opinion_per_collection_result_evaluator(
            test_predict_filepath=predict_filename,
            etalon_samples_filepath=etalon_filename,
            test_samples_filepath=test_filename,
            label_scaler=PosNegNeuRelationsLabelScaler())

        TestEval.__print_result(result)

    def test_opinion_eval(self):

        predict_filename, test_filename, etalon_filename = self.__create_data()

        result = opinions_per_document_two_class_result_evaluation(
            test_predict_filepath=predict_filename,
            etalon_samples_filepath=etalon_filename,
            test_samples_filepath=test_filename,
            label_scaler=PosNegNeuRelationsLabelScaler())

        TestEval.__print_result(result)


if __name__ == '__main__':
    unittest.main()
