from os.path import exists

import numpy as np
from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import BaseSampleStorageView
from tqdm import tqdm

from labels.scaler import PosNegNeuRelationsLabelScaler


def calculate_totat_samples_count_per_label(test_predict_filepath,
                                            label_scaler=PosNegNeuRelationsLabelScaler()):
    assert(isinstance(test_predict_filepath, str))

    if not exists(test_predict_filepath):
        raise FileNotFoundError(test_predict_filepath)

    predict_view = BaseSampleStorageView(storage=BaseRowsStorage.from_tsv(filepath=test_predict_filepath),
                                         row_ids_provider=MultipleIDProvider())

    test_linked_iter = predict_view.iter_rows_linked_by_text_opinions()
    labels_stat = {}
    for linkage in tqdm(test_linked_iter):
        for row in linkage:
            uint_labels = [int(row[str(c)]) for c in range(label_scaler.classes_count())]
            label = int(np.argmax(uint_labels))
            if label not in labels_stat:
                labels_stat[label] = 0
            labels_stat[label] += 1

    return labels_stat
