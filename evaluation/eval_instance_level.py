from arekit.common.data.input.readers.tsv import TsvReader
from arekit.common.data.storages.base import BaseRowsStorage
from tqdm import tqdm
from collections import OrderedDict
from os.path import exists

from arekit.common.data.views.samples import LinkedSamplesStorageView
from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.evaluation.comparators.text_opinions import TextOpinionBasedComparator
from arekit.common.evaluation.evaluators.modes import EvaluationModes
from arekit.common.evaluation.pairs.single import SingleDocumentDataPairsToCompare

from SentiNEREL.labels.scaler import PosNegNeuRelationsLabelScaler
from evaluation.utils import assign_labels, row_to_context_opinion, create_evaluator, create_filter_labels_func


def extract_context_opinions_by_row_id(view, storage, label_scaler, no_label):
    """ Reading data from tsv-gz via storage view
        returns: dict
            (row_id, text_opinion)
    """
    assert(isinstance(view, LinkedSamplesStorageView))
    assert(isinstance(storage, BaseRowsStorage))

    context_opinions_by_row_id = OrderedDict()
    for linkage in tqdm(view.iter_from_storage(storage)):
        for row in linkage:
            context_opinions_by_row_id[row["id"]] = row_to_context_opinion(
                row=row, label_scaler=label_scaler, default_label=no_label)

    return context_opinions_by_row_id


def text_opinion_per_collection_result_evaluator(
        test_predict_filepath, etalon_samples_filepath, test_samples_filepath,
        evaluator_type="two_class",
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
    assert(isinstance(evaluator_type, str))

    if not exists(test_predict_filepath):
        raise FileNotFoundError(test_predict_filepath)

    if not exists(etalon_samples_filepath):
        raise FileNotFoundError(etalon_samples_filepath)

    if not exists(test_samples_filepath):
        raise FileNotFoundError(test_samples_filepath)

    # TODO. #363 нужно переделать API на передачу просто меток, игнорируемых меток.
    no_label = label_scaler.uint_to_label(0)

    reader = TsvReader()
    etalon_samples_storage = reader.read(target=etalon_samples_filepath)
    test_samples_storage = reader.read(target=test_samples_filepath)
    predict_samples_storage = reader.read(target=test_predict_filepath)

    view = LinkedSamplesStorageView(row_ids_provider=MultipleIDProvider())

    # Setup filter
    filter_context_opinion_func = create_filter_labels_func(
        evaluator_type=evaluator_type,
        get_label_func=lambda context_opinion: context_opinion.Sentiment,
        no_label=no_label)

    # Reading collection through storage views.
    etalon_context_opinions_by_row_id = extract_context_opinions_by_row_id(
        view=view, storage=etalon_samples_storage, label_scaler=label_scaler, no_label=no_label)
    test_context_opinions_by_row_id = extract_context_opinions_by_row_id(
        view=view, storage=test_samples_storage, label_scaler=label_scaler, no_label=no_label)
    assign_labels(view=view,
                  storage=predict_samples_storage,
                  text_opinions=test_context_opinions_by_row_id.values(),
                  row_id_to_context_opin_id_func=lambda row_id:
                      test_context_opinions_by_row_id[row_id].Tag,
                  label_scaler=label_scaler)

    eee = len(etalon_context_opinions_by_row_id)
    ttt = len(test_context_opinions_by_row_id)
    print(eee)
    print(ttt)

    # Remove the one with NoLabel instance.
    test_context_opinions_by_row_id = {
        row_id: text_opinion for row_id, text_opinion in test_context_opinions_by_row_id.items()
        if filter_context_opinion_func(text_opinion)
    }

    etalon_context_opinions_by_row_id = {
        row_id: text_opinion for row_id, text_opinion in etalon_context_opinions_by_row_id.items()
        if filter_context_opinion_func(text_opinion)
    }

    eee = len(etalon_context_opinions_by_row_id)
    ttt = len(test_context_opinions_by_row_id)
    print(eee)
    print(ttt)

    # Composing evaluator.
    evaluator = create_evaluator(evaluator_type=evaluator_type,
                                 comparator=TextOpinionBasedComparator(eval_mode=EvaluationModes.Extraction),
                                 # TODO. #363 нужно переделать API на передачу просто меток, игнорируемых меток.
                                 uint_labels=[1, 2, 0],
                                 get_item_label_func=lambda text_opinion: text_opinion.Sentiment,
                                 label_scaler=label_scaler)

    # evaluate every document.
    cmp_pair = SingleDocumentDataPairsToCompare(etalon_data=list(etalon_context_opinions_by_row_id.values()),
                                                test_data=list(test_context_opinions_by_row_id.values()))
    result = evaluator.evaluate(cmp_pairs=[cmp_pair])

    return result
