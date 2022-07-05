import numpy as np
from tqdm import tqdm
from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.text_opinions.base import TextOpinion


def assign_labels(filename, text_opinions, row_id_to_text_opin_id_func, label_scaler):
    """ Назначение меток с результата разметки на TextOpinion соответствующего множества.
    """
    assert(callable(row_id_to_text_opin_id_func))

    text_opinons_by_id = {}
    for text_opinion in text_opinions:
        assert (isinstance(text_opinion, TextOpinion))
        text_opinons_by_id[text_opinion.TextOpinionID] = text_opinion

    predict_view = BaseSampleStorageView(storage=BaseRowsStorage.from_tsv(filepath=filename),
                                         row_ids_provider=MultipleIDProvider())
    test_linked_iter = predict_view.iter_rows_linked_by_text_opinions()
    for linkage in tqdm(test_linked_iter):
        for row in linkage:
            text_opinion_id = row_id_to_text_opin_id_func(row["id"])
            text_opinion = text_opinons_by_id[text_opinion_id]
            uint_labels = [int(row[str(c)]) for c in range(label_scaler.classes_count())]
            text_opinion.set_label(label_scaler.uint_to_label(int(np.argmax(uint_labels))))
