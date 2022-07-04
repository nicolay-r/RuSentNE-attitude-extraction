import unittest
from collections import OrderedDict

import numpy as np
from os.path import join, exists, dirname, realpath

from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.evaluation.comparators.text_opinions import TextOpinionBasedComparator
from arekit.common.evaluation.evaluators.base import BaseEvaluator
from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.evaluation.pairs.single import SingleDocumentDataPairsToCompare
from arekit.common.labels.base import NoLabel
from arekit.common.text_opinions.base import TextOpinion
from arekit.common.utils import progress_bar_iter
from arekit.contrib.utils.evaluation.results.three_class import ThreeClassEvalResult
from tqdm import tqdm

from labels.scaler import PosNegNeuRelationsLabelScaler


class ThreeClassTextOpinionEvaluator(BaseEvaluator):

    def __init__(self, comparator, label1, label2, no_label):
        super(ThreeClassTextOpinionEvaluator, self).__init__(comparator)

        self.__label1 = label1
        self.__label2 = label2
        self.__no_label = no_label

    def _create_eval_result(self):
        return ThreeClassEvalResult(label1=self.__label1, label2=self.__label2, no_label=self.__no_label,
                                    get_item_label_func=lambda text_opinion: text_opinion.Sentiment)


class TestEval(unittest.TestCase):

    @staticmethod
    def __extract_text_opinions(filename, label_scaler):
        """ Reading data from tsv-gz via storage view
            returns: dict
                (row_id, text_opinion)
        """
        etalon_view = BaseSampleStorageView(storage=BaseRowsStorage.from_tsv(filepath=filename),
                                            row_ids_provider=MultipleIDProvider())
        etalon_linked_iter = etalon_view.iter_rows_linked_by_text_opinions()
        opinions_by_row_id = OrderedDict()
        for linkage in tqdm(etalon_linked_iter):
            for row in linkage:

                uint_label = int(row["label"]) if "label" in row \
                    else label_scaler.label_to_uint(NoLabel())

                text_opinion = TextOpinion(
                    doc_id=int(row["doc_id"]),
                    text_opinion_id=None,
                    source_id=int(row["s_ind"]),
                    target_id=int(row["t_ind"]),
                    owner=None,
                    label=label_scaler.uint_to_label(uint_label))

                tid = TextOpinionBasedComparator.text_opinion_to_id(text_opinion)
                text_opinion.set_text_opinion_id(tid)

                opinions_by_row_id[row["id"]] = text_opinion

        return opinions_by_row_id

    @staticmethod
    def __assign_labels(filename, text_opinions, row_id_to_text_opin_id_func, label_scaler):
        assert(callable(row_id_to_text_opin_id_func))

        text_opinons_by_id = {}
        for text_opinion in text_opinions:
            assert(isinstance(text_opinion, TextOpinion))
            text_opinons_by_id[text_opinion.TextOpinionID] = text_opinion

        predict_view = BaseSampleStorageView(storage=BaseRowsStorage.from_tsv(filepath=filename),
                                             row_ids_provider=MultipleIDProvider())
        test_linked_iter = predict_view.iter_rows_linked_by_text_opinions()
        test_opinions = []
        for linkage in tqdm(test_linked_iter):
            for row in linkage:
                text_opinion_id = row_id_to_text_opin_id_func(row["id"])
                text_opinion = text_opinons_by_id[text_opinion_id]
                uint_labels = [int(row[str(c)]) for c in range(label_scaler.classes_count())]
                text_opinion.set_label(label_scaler.uint_to_label(int(np.argmax(uint_labels))))
                test_opinions.append(text_opinion)

    def test(self):
        """ We adopt AREkit API for this and relying on the following example:
            https://github.com/nicolay-r/AREkit/blob/master/tests/contrib/utils/test_eval.py
        """

        current_dir = dirname(realpath(__file__))
        output_dir = join(current_dir, "_out")
        label_scaler = PosNegNeuRelationsLabelScaler()

        source_filename = "predict.tsv.gz"
        serialize_dir = "serialize-nn_3l"

        predict_filename = join(output_dir, serialize_dir, source_filename)
        test_filename = join(output_dir, serialize_dir, "sample-test-0.tsv.gz")
        etalon_filename = join(output_dir, serialize_dir, "sample-etalon-0.tsv.gz")

        if not exists(predict_filename):
            raise FileNotFoundError(predict_filename)

        if not exists(etalon_filename):
            raise FileNotFoundError(etalon_filename)

        ###########################################
        # Reading collection through storage views.
        ###########################################
        etalon_opins_by_row_id = TestEval.__extract_text_opinions(filename=etalon_filename, label_scaler=label_scaler)
        test_opins_by_row_id = TestEval.__extract_text_opinions(filename=test_filename, label_scaler=label_scaler)
        TestEval.__assign_labels(filename=predict_filename,
                                 text_opinions=test_opins_by_row_id.values(),
                                 row_id_to_text_opin_id_func=lambda row_id: test_opins_by_row_id[row_id].TextOpinionID,
                                 label_scaler=label_scaler)

        ########################
        # Composing comparators.
        ########################
        evaluator = ThreeClassTextOpinionEvaluator(
            comparator=TextOpinionBasedComparator(eval_mode=EvaluationModes.Extraction),
            label1=label_scaler.uint_to_label(1),
            label2=label_scaler.uint_to_label(2),
            no_label=label_scaler.uint_to_label(0))

        # evaluate every document.
        cmp_pair = SingleDocumentDataPairsToCompare(etalon_data=list(etalon_opins_by_row_id.values()),
                                                    test_data=list(test_opins_by_row_id.values()))
        logged_cmp_pairs_it = progress_bar_iter([cmp_pair], desc="Evaluate", unit='pairs')
        result = evaluator.evaluate(cmp_pairs=logged_cmp_pairs_it)

        print(result)

        result.calculate()

        print(result.TotalResult)
        print(result.get_result_by_metric(ThreeClassEvalResult.C_F1))
        print(result.get_result_by_metric(ThreeClassEvalResult.C_F1_POS))
        print(result.get_result_by_metric(ThreeClassEvalResult.C_F1_NEG))


if __name__ == '__main__':
    unittest.main()
