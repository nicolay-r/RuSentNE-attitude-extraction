from tqdm import tqdm
from collections import OrderedDict
from os.path import exists

from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.evaluation.comparators.text_opinions import TextOpinionBasedComparator
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.opinions.base import Opinion
from arekit.common.text_opinions.base import TextOpinion
from labels.scaler import PosNegNeuRelationsLabelScaler


def __row_to_opinion(row, label_scaler):
    """ составление Opinion из ряда данных sample.
    """

    uint_label = int(row["label"]) if "label" in row \
        else label_scaler.label_to_uint(NoLabel())

    source_id = int(row["s_ind"])
    target_id = int(row["t_ind"])
    entity_values = row["entity_values"].split(',')

    return Opinion(source_value=entity_values[source_id],
                   target_value=entity_values[target_id],
                   sentiment=label_scaler.uint_to_label(uint_label))


def __row_to_text_opinion(row, label_scaler):
    """ Чтение text_opinion из ряда данных sample.
    """
    assert(isinstance(label_scaler, BaseLabelScaler))

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

    return text_opinion


def __extract_text_opinions_from_test(test_view, label_scaler):
    """
        return: dict { tid: TextOpinion },
        where:  tid -- is a TextOpininID
    """
    assert(isinstance(test_view, BaseSampleStorageView))
    assert(isinstance(label_scaler, BaseLabelScaler))

    text_opinion_by_id = OrderedDict()
    for linkage in tqdm(test_view.iter_rows_linked_by_text_opinions()):
        for row in linkage:
            text_opinion = __row_to_text_opinion(row, label_scaler)
            text_opinion_by_id[text_opinion.TextOpinionID] = text_opinion

    return text_opinion_by_id


def __gather_opinion_and_group_ids_from_etalon(etalon_view, label_scaler):
    """ Из связок берем только первое отношение и считаем его Opinion.
        А также группируем TextOpinons относительно первого id.
    """
    opinion_by_row_id = OrderedDict()
    text_opinion_ids_by_row_id = OrderedDict()
    for linkage in tqdm(etalon_view.iter_rows_linked_by_text_opinions()):
        first_row = linkage[0]
        first_row_id = first_row["id"]
        opinion_by_row_id[first_row_id] = __row_to_opinion(first_row, label_scaler)
        text_opinion_ids_by_row_id[first_row_id] = [__row_to_text_opinion(row, label_scaler).TextOpinionID
                                                    for row in linkage]

    return opinion_by_row_id, text_opinion_ids_by_row_id


def __compose_test_opinions(etalon_opinions_by_row_id, etalon_text_opinion_ids_by_row_id,
                            test_opinions_by_id, label_scaler):

    used = set()

    test_opinions = []
    for row_id, etalon_opinion in etalon_opinions_by_row_id.items():
        tid_list = etalon_text_opinion_ids_by_row_id[row_id]
        labels = [label_scaler.label_to_int(test_opinions_by_id[tid].Sentiment) for tid in tid_list]

        # используем метод голосования
        vote_label = sum(labels)
        if vote_label < 0:
            vote_label = -1
        if vote_label > 0:
            vote_label = 1

        test_opinion = Opinion(source_value=etalon_opinion.SourceValue,
                               target_value=etalon_opinion.TargetValue,
                               sentiment=label_scaler.int_to_label(vote_label))

        test_opinions.append(test_opinion)

        # отмечаем как used.
        for tid in tid_list:
            used.add(test_opinions_by_id[tid])

    # TODO. Осталось учесть тех, что нет в Etalon.

    return test_opinions


def opinions_per_document_result_evaluation(
        predict_filename, etalon_samples_filepath, test_samples_filepath,
        label_scaler=PosNegNeuRelationsLabelScaler()):
    """ Подокументное вычисление результатов разметки отношений типа Opninon (пар на уровне документа)
    """
    assert(isinstance(predict_filename, str))
    assert(isinstance(etalon_samples_filepath, str))
    assert(isinstance(test_samples_filepath, str))

    if not exists(predict_filename):
        raise FileNotFoundError(predict_filename)

    if not exists(etalon_samples_filepath):
        raise FileNotFoundError(etalon_samples_filepath)

    if not exists(test_samples_filepath):
        raise FileNotFoundError(test_samples_filepath)

    test_view = BaseSampleStorageView(storage=BaseRowsStorage.from_tsv(filepath=test_samples_filepath),
                                      row_ids_provider=MultipleIDProvider())
    etalon_view = BaseSampleStorageView(storage=BaseRowsStorage.from_tsv(filepath=etalon_samples_filepath),
                                        row_ids_provider=MultipleIDProvider())

    test_opinions_by_id = __extract_text_opinions_from_test(test_view=test_view, label_scaler=label_scaler)
    etalon_opinions_by_row_id, etalon_text_opinion_ids_by_row_id = __gather_opinion_and_group_ids_from_etalon(
        etalon_view=etalon_view, label_scaler=label_scaler)
    test_opinions = __compose_test_opinions(etalon_opinions_by_row_id=etalon_opinions_by_row_id,
                                            etalon_text_opinion_ids_by_row_id=etalon_text_opinion_ids_by_row_id,
                                            test_opinions_by_id=test_opinions_by_id,
                                            label_scaler=label_scaler)

    return