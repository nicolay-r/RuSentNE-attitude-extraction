from collections import OrderedDict
from itertools import chain
from os.path import exists

from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.evaluation.comparators.text_opinions import TextOpinionBasedComparator
from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.evaluation.utils import OpinionCollectionsToCompareUtils
from arekit.common.utils import progress_bar_iter
from arekit.contrib.utils.evaluation.evaluators.two_class import TwoClassEvaluator
from tqdm import tqdm

from evaluation.instance_level import extract_text_opinions_by_row_id
from evaluation.utils import assign_labels, row_to_text_opinion
from labels.scaler import PosNegNeuRelationsLabelScaler


def __group_text_opinions_by_doc_id(view):
    text_opinions_by_doc_id = OrderedDict()
    for linkage in tqdm(view.iter_rows_linked_by_text_opinions()):
        for row in linkage:
            doc_id = row["doc_id"]
            if doc_id not in text_opinions_by_doc_id:
                text_opinions_by_doc_id[doc_id] = []
            text_opinions_by_doc_id[doc_id].append(row["id"])

    return text_opinions_by_doc_id


def text_opinion_per_document_two_class_result_evaluator(
        test_predict_filepath, etalon_samples_filepath, test_samples_filepath,
        label_scaler=PosNegNeuRelationsLabelScaler()):
    assert(isinstance(test_predict_filepath, str))
    assert(isinstance(etalon_samples_filepath, str))
    assert(isinstance(test_samples_filepath, str))

    if not exists(test_predict_filepath):
        raise FileNotFoundError(test_predict_filepath)

    if not exists(etalon_samples_filepath):
        raise FileNotFoundError(etalon_samples_filepath)

    if not exists(test_samples_filepath):
        raise FileNotFoundError(test_samples_filepath)

    no_label = label_scaler.uint_to_label(0)

    # Setup views.
    etalon_view = BaseSampleStorageView(storage=BaseRowsStorage.from_tsv(filepath=etalon_samples_filepath),
                                        row_ids_provider=MultipleIDProvider())
    test_view = BaseSampleStorageView(storage=BaseRowsStorage.from_tsv(filepath=test_samples_filepath),
                                      row_ids_provider=MultipleIDProvider())
    predict_view = BaseSampleStorageView(storage=BaseRowsStorage.from_tsv(filepath=test_predict_filepath),
                                         row_ids_provider=MultipleIDProvider())

    # Reading collection through storage views.
    etalon_text_opinions_by_row_id = extract_text_opinions_by_row_id(
        view=etalon_view, label_scaler=label_scaler, no_label=no_label)
    test_text_opinions_by_row_id = extract_text_opinions_by_row_id(
        view=test_view, label_scaler=label_scaler, no_label=no_label)

    # Gathering them by doc_id.
    etalon_row_ids_by_doc_id = __group_text_opinions_by_doc_id(view=etalon_view)
    test_row_ids_by_doc_id = __group_text_opinions_by_doc_id(view=test_view)

    assign_labels(predict_view=predict_view,
                  text_opinions=test_text_opinions_by_row_id.values(),
                  row_id_to_text_opin_id_func=lambda row_id: test_text_opinions_by_row_id[row_id].TextOpinionID,
                  label_scaler=label_scaler)

    doc_ids = sorted(list(set(chain(test_row_ids_by_doc_id.keys(), etalon_row_ids_by_doc_id.keys()))))

    cmp_pairs_iter = OpinionCollectionsToCompareUtils.iter_comparable_collections(
        doc_ids=[int(doc_id) for doc_id in doc_ids],
        read_etalon_collection_func=lambda doc_id:
            [etalon_text_opinions_by_row_id[row_id] for row_id in etalon_row_ids_by_doc_id[doc_id]]
            if doc_id in etalon_row_ids_by_doc_id else [],
        read_test_collection_func=lambda doc_id:
            # Удаляем среди перечня те отношения, у которых оценка NoLabel.
            [test_text_opinions_by_row_id[row_id] for row_id in test_row_ids_by_doc_id[doc_id]
             if test_text_opinions_by_row_id[row_id].Sentiment != no_label]
            if doc_id in test_row_ids_by_doc_id else [])

    # Composing evaluator.
    evaluator = TwoClassEvaluator(
        comparator=TextOpinionBasedComparator(eval_mode=EvaluationModes.Extraction),
        label1=label_scaler.uint_to_label(1),
        label2=label_scaler.uint_to_label(2),
        get_item_label_func=lambda opinion: opinion.Sentiment)

    # evaluate every document.
    logged_cmp_pairs_it = progress_bar_iter(cmp_pairs_iter, desc="Evaluate", unit='pairs')
    result = evaluator.evaluate(cmp_pairs=logged_cmp_pairs_it)

    return result
