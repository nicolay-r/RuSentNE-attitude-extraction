import numpy as np
from arekit.common.evaluation.comparators.text_opinions import TextOpinionBasedComparator
from arekit.common.labels.base import Label
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.opinions.base import Opinion
from tqdm import tqdm
from arekit.common.text_opinions.base import TextOpinion


def assign_labels(predict_view, text_opinions, row_id_to_text_opin_id_func, label_scaler):
    """ Назначение меток с результата разметки на TextOpinion соответствующего множества.
    """
    assert(callable(row_id_to_text_opin_id_func))

    text_opinons_by_id = {}
    for text_opinion in text_opinions:
        assert (isinstance(text_opinion, TextOpinion))
        text_opinons_by_id[text_opinion.TextOpinionID] = text_opinion

    test_linked_iter = predict_view.iter_rows_linked_by_text_opinions()
    for linkage in tqdm(test_linked_iter):
        for row in linkage:
            text_opinion_id = row_id_to_text_opin_id_func(row["id"])
            text_opinion = text_opinons_by_id[text_opinion_id]
            uint_labels = [int(row[str(c)]) for c in range(label_scaler.classes_count())]
            text_opinion.set_label(label_scaler.uint_to_label(int(np.argmax(uint_labels))))


def row_to_text_opinion(row, label_scaler, default_label):
    """ Чтение text_opinion из ряда данных sample.
        мы также создаем уникальный идентификатор TextOpinion
        на основе индекса предложения документа и индексов сущностей в нем.
    """
    assert(isinstance(label_scaler, BaseLabelScaler))
    assert(isinstance(default_label, Label))

    uint_label = int(row["label"]) if "label" in row \
        else label_scaler.label_to_uint(default_label)

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
