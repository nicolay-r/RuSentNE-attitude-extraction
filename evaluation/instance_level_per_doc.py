from tqdm import tqdm
from collections import OrderedDict
from itertools import chain
from os.path import exists

from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.evaluation.comparators.text_opinions import TextOpinionBasedComparator
from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.utils import progress_bar_iter
from arekit.contrib.utils.evaluation.iterators import DataPairsIterators

from evaluation.factory import create_filter_labels_func, create_evaluator
from evaluation.instance_level import extract_context_opinions_by_row_id
from evaluation.utils import assign_labels
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
        evaluator_type="two_class", label_scaler=PosNegNeuRelationsLabelScaler()):
    """ Выполнение оценки на уровне разметки экземпляров;
        оценка вычисляется по каждому документу в отдельности, а
        затем подсчитывается среднее значение.

        # TODO. #363 нужно переделать API на передачу просто меток, игнорируемых меток.
    """
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
    etalon_context_opinions_by_row_id = extract_context_opinions_by_row_id(
        view=etalon_view, label_scaler=label_scaler, no_label=no_label)
    test_context_opinions_by_row_id = extract_context_opinions_by_row_id(
        view=test_view, label_scaler=label_scaler, no_label=no_label)

    # Gathering them by doc_id.
    etalon_row_ids_by_doc_id = __group_text_opinions_by_doc_id(view=etalon_view)
    test_row_ids_by_doc_id = __group_text_opinions_by_doc_id(view=test_view)

    assign_labels(predict_view=predict_view,
                  text_opinions=test_context_opinions_by_row_id.values(),
                  row_id_to_context_opin_id_func=lambda row_id: test_context_opinions_by_row_id[row_id].Tag,
                  label_scaler=label_scaler)

    doc_ids = sorted(list(set(chain(test_row_ids_by_doc_id.keys(), etalon_row_ids_by_doc_id.keys()))))

    # Setup filter
    filter_text_opinion_func = create_filter_labels_func(
        evaluator_type=evaluator_type,
        get_label_func=lambda text_opinion: text_opinion.Sentiment,
        no_label=no_label)

    cmp_pairs_iter = DataPairsIterators.iter_func_based_collections(
        doc_ids=[int(doc_id) for doc_id in doc_ids],
        read_etalon_collection_func=lambda doc_id:
            [etalon_context_opinions_by_row_id[row_id] for row_id in etalon_row_ids_by_doc_id[doc_id]
             if filter_text_opinion_func(etalon_context_opinions_by_row_id[row_id])]
            if doc_id in etalon_row_ids_by_doc_id else [],
        read_test_collection_func=lambda doc_id:
            # Удаляем среди перечня те отношения, у которых оценка NoLabel.
            [test_context_opinions_by_row_id[row_id] for row_id in test_row_ids_by_doc_id[doc_id]
             if filter_text_opinion_func(test_context_opinions_by_row_id[row_id])]
            if doc_id in test_row_ids_by_doc_id else [])

    # Composing evaluator.
    evaluator = create_evaluator(evaluator_type=evaluator_type,
                                 comparator=TextOpinionBasedComparator(eval_mode=EvaluationModes.Extraction),
                                 uint_labels=[1, 2, 0],
                                 get_item_label_func=lambda opinion: opinion.Sentiment,
                                 label_scaler=label_scaler)

    # evaluate every document.
    logged_cmp_pairs_it = progress_bar_iter(cmp_pairs_iter, desc="Evaluate", unit='pairs')
    result = evaluator.evaluate(cmp_pairs=logged_cmp_pairs_it)

    return result
