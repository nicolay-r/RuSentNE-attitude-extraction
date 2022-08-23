import numpy as np
from tqdm import tqdm
from os.path import exists

from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import LinkedSamplesStorageView

from SentiNEREL.labels.scaler import PosNegNeuRelationsLabelScaler


def calculate_predicted_count_per_label(test_predict_filepath,
                                        label_scaler=PosNegNeuRelationsLabelScaler()):
    assert(isinstance(test_predict_filepath, str))

    if not exists(test_predict_filepath):
        raise FileNotFoundError(test_predict_filepath)

    predict_linked_view = LinkedSamplesStorageView(
        storage=BaseRowsStorage.from_tsv(filepath=test_predict_filepath),
        row_ids_provider=MultipleIDProvider())

    test_linked_iter = predict_linked_view
    labels_stat = {}
    for linkage in tqdm(test_linked_iter):
        for row in linkage:
            uint_labels = [int(row[str(c)]) for c in range(label_scaler.classes_count())]
            label = int(np.argmax(uint_labels))
            if label not in labels_stat:
                labels_stat[label] = 0
            labels_stat[label] += 1

    return labels_stat


def calculate_samples_count_per_label(samples_filepath, no_label_uint):
    assert(isinstance(samples_filepath, str))

    if not exists(samples_filepath):
        raise FileNotFoundError(samples_filepath)

    predict_linked_view = LinkedSamplesStorageView(
        storage=BaseRowsStorage.from_tsv(filepath=samples_filepath),
        row_ids_provider=MultipleIDProvider())

    test_linked_iter = predict_linked_view
    labels_stat = {}
    used_row_ids = set()

    for linkage in tqdm(test_linked_iter):
        for row in linkage:

            if row["id"] in used_row_ids:
                continue

            uint_label = int(row["label"]) if "label" in row else no_label_uint

            if uint_label not in labels_stat:
                labels_stat[uint_label] = 0
            labels_stat[uint_label] += 1

            used_row_ids.add(row["id"])

    return labels_stat
