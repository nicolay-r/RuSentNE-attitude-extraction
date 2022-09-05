from collections import Iterable
from itertools import chain

import numpy as np
from arekit.common.data import const
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import LinkedSamplesStorageView
from arekit.common.evaluation.comparators.text_opinions import TextOpinionBasedComparator
from arekit.common.evaluation.context_opinion import ContextOpinion
from arekit.common.labels.base import Label
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.opinions.base import Opinion
from arekit.contrib.utils.evaluation.evaluators.three_class import ThreeClassEvaluator
from arekit.contrib.utils.evaluation.evaluators.two_class import TwoClassEvaluator
from tqdm import tqdm


def assign_labels(view, storage, text_opinions, row_id_to_context_opin_id_func, label_scaler):
    """ Назначение меток с результата разметки на TextOpinion соответствующего множества.
    """
    assert(isinstance(view, LinkedSamplesStorageView))
    assert(isinstance(storage, BaseRowsStorage))
    assert(callable(row_id_to_context_opin_id_func))

    text_opinons_by_id = {}
    for context_opinion in text_opinions:
        assert (isinstance(context_opinion, ContextOpinion))
        text_opinons_by_id[context_opinion.Tag] = context_opinion

    for linkage in tqdm(view.iter_from_storage(storage)):
        for row in linkage:
            text_opinion_id = row_id_to_context_opin_id_func(row[const.ID])
            context_opinion = text_opinons_by_id[text_opinion_id]
            uint_labels = [int(row[str(c)]) for c in range(label_scaler.classes_count())]
            context_opinion.set_label(label_scaler.uint_to_label(int(np.argmax(uint_labels))))


def row_to_context_opinion(row, label_scaler, default_label):
    """ Чтение text_opinion из ряда данных sample.
        мы также создаем уникальный идентификатор TextOpinion
        на основе индекса предложения документа и индексов сущностей в нем.
    """
    assert(isinstance(label_scaler, BaseLabelScaler))
    assert(isinstance(default_label, Label))

    uint_label = int(row["label"]) if "label" in row \
        else label_scaler.label_to_uint(default_label)

    context_opinion = ContextOpinion(doc_id=int(row["doc_id"]),
                                     source_id=int(row["s_ind"]),
                                     target_id=int(row["t_ind"]),
                                     label=label_scaler.uint_to_label(uint_label),
                                     context_id=row["sent_ind"])    # for now, it is just a single sentence

    context_opinion.set_tag(TextOpinionBasedComparator.context_opinion_to_id(context_opinion))

    return context_opinion


def row_to_opinion(row, label_scaler, default_label):
    """ составление Opinion из ряда данных sample.
    """
    assert(isinstance(default_label, Label))

    uint_label = int(row["label"]) if "label" in row \
        else label_scaler.label_to_uint(default_label)

    source_index = int(row["s_ind"])
    target_index = int(row["t_ind"])
    entities = [int(e) for e in row["entities"].split(',')]
    entity_values = row["entity_values"].split(',')

    return Opinion(source_value=entity_values[entities.index(source_index)],
                   target_value=entity_values[entities.index(target_index)],
                   sentiment=label_scaler.uint_to_label(uint_label))


def create_evaluator(evaluator_type, comparator, label_scaler, get_item_label_func, uint_labels):
    """ TODO: #363
        https://github.com/nicolay-r/AREkit/issues/363
        This should bere removed, since we consider a MulticlassEvaluator.
        This is now limited to 2 and 3.
    """
    assert(isinstance(evaluator_type, str))
    assert(isinstance(uint_labels, list))

    if evaluator_type == "two_class":
        return TwoClassEvaluator(
            comparator=comparator,
            label1=label_scaler.uint_to_label(uint_labels[0]),
            label2=label_scaler.uint_to_label(uint_labels[1]),
            get_item_label_func=get_item_label_func)

    if evaluator_type == "three_class":
        return ThreeClassEvaluator(
            comparator=comparator,
            label1=label_scaler.uint_to_label(uint_labels[0]),
            label2=label_scaler.uint_to_label(uint_labels[1]),
            label3=label_scaler.uint_to_label(uint_labels[2]),
            get_item_label_func=get_item_label_func)


def create_filter_labels_func(evaluator_type, get_label_func, no_label):
    """ TODO: #363
        https://github.com/nicolay-r/AREkit/issues/363
        provide just labels that should be ignored instead, once #363 will providee Multiclass Evaluator.
        This is now limited to 2 and 3.
    """
    assert(callable(get_label_func))

    if evaluator_type == "two_class":
        return lambda item: get_label_func(item) != no_label
    if evaluator_type == "three_class":
        return lambda item: True


def select_doc_ids(doc_ids_mode, test_doc_ids, etalon_doc_ids):
    """ Правила выбора документов для выполнения оценки.
    """
    assert(doc_ids_mode in ["etalon", "joined"])
    assert(isinstance(test_doc_ids, Iterable))
    assert(isinstance(etalon_doc_ids, Iterable))

    data = None

    if doc_ids_mode == "etalon":
        # Рассматриваем только те, которые присутствуют в эталонном множестве.
        data = sorted(list(etalon_doc_ids))
    if doc_ids_mode == "joined":
        # Рассматриваем все документы, включая те, в которых в эталонной разметке нет.
        data = set(chain(test_doc_ids, etalon_doc_ids))

    return sorted(list(data))
