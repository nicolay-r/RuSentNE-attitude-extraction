from collections import OrderedDict
from os.path import exists

from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.evaluation.comparators.text_opinions import TextOpinionBasedComparator
from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.evaluation.pairs.single import SingleDocumentDataPairsToCompare
from arekit.contrib.utils.evaluation.evaluators.two_class import TwoClassEvaluator
from tqdm import tqdm

from evaluation.utils import assign_labels, row_to_text_opinion
from labels.scaler import PosNegNeuRelationsLabelScaler


def extract_text_opinions_by_row_id(view, label_scaler, no_label):
    """ Reading data from tsv-gz via storage view
        returns: dict
            (row_id, text_opinion)
    """
    text_opinions_by_row_id = OrderedDict()
    for linkage in tqdm(view.iter_rows_linked_by_text_opinions()):
        for row in linkage:
            text_opinions_by_row_id[row["id"]] = row_to_text_opinion(
                row=row, label_scaler=label_scaler, default_label=no_label)

    return text_opinions_by_row_id


def text_opinion_per_collection_two_class_result_evaluator(
        test_predict_filepath, etalon_samples_filepath, test_samples_filepath,
        label_scaler=PosNegNeuRelationsLabelScaler()):
    """ Single-document like (whole collection) evaluator.
        Considering text_opinion instances as items for comparison.

        Оценка выполняется на уровне контекстных отношений.
        Учет по документам не идет, т.е. предполагается
        целая коллекция как один огромный документ.
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
    etalon_text_opinions_by_row_id = extract_text_opinions_by_row_id(
        view=etalon_view, label_scaler=label_scaler, no_label=no_label)
    test_text_opinions_by_row_id = extract_text_opinions_by_row_id(
        view=test_view, label_scaler=label_scaler, no_label=no_label)
    assign_labels(predict_view=predict_view,
                  text_opinions=test_text_opinions_by_row_id.values(),
                  row_id_to_text_opin_id_func=lambda row_id: test_text_opinions_by_row_id[row_id].TextOpinionID,
                  label_scaler=label_scaler)

    # Remove the one with NoLabel instance.
    test_text_opinions_by_row_id = {row_id: text_opinion for row_id, text_opinion in test_text_opinions_by_row_id.items()
                                    if text_opinion.Sentiment != no_label}

    # Composing evaluator.
    evaluator = TwoClassEvaluator(
        comparator=TextOpinionBasedComparator(eval_mode=EvaluationModes.Extraction),
        label1=label_scaler.uint_to_label(1),
        label2=label_scaler.uint_to_label(2),
        get_item_label_func=lambda text_opinion: text_opinion.Sentiment)

    # evaluate every document.
    cmp_pair = SingleDocumentDataPairsToCompare(etalon_data=list(etalon_text_opinions_by_row_id.values()),
                                                test_data=list(test_text_opinions_by_row_id.values()))
    result = evaluator.evaluate(cmp_pairs=[cmp_pair])

    return result
