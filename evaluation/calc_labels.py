import numpy as np
from tqdm import tqdm
from os.path import exists
from arekit.common.data import const
from arekit.common.data.input.readers.tsv import TsvReader
from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.views.samples import LinkedSamplesStorageView

from SentiNEREL.labels.scaler import PosNegNeuRelationsLabelScaler


def calculate_predicted_count_per_label(test_predict_filepath,
                                        reader=TsvReader(),
                                        label_scaler=PosNegNeuRelationsLabelScaler()):
    assert(isinstance(test_predict_filepath, str))

    if not exists(test_predict_filepath):
        raise FileNotFoundError(test_predict_filepath)

    view = LinkedSamplesStorageView(MultipleIDProvider())
    storage = reader.read(test_predict_filepath)
    labels_stat = {}

    for linkage in tqdm(view.iter_from_storage(storage)):
        for row in linkage:
            uint_labels = [int(row[str(c)]) for c in range(label_scaler.classes_count())]
            label = int(np.argmax(uint_labels))
            if label not in labels_stat:
                labels_stat[label] = 0
            labels_stat[label] += 1

    return labels_stat


def calculate_samples_count_per_label(samples_filepath, no_label_uint, reader=TsvReader()):
    assert(isinstance(samples_filepath, str))

    if not exists(samples_filepath):
        raise FileNotFoundError(samples_filepath)

    used_row_ids = set()

    storage = reader.read(samples_filepath)
    predict_linked_view = LinkedSamplesStorageView(MultipleIDProvider())
    labels_stat = {}

    for linkage in tqdm(predict_linked_view.iter_from_storage(storage)):
        for row in linkage:

            if row[const.ID] in used_row_ids:
                continue

            uint_label = int(row[const.LABEL]) if const.LABEL in row else no_label_uint

            if uint_label not in labels_stat:
                labels_stat[uint_label] = 0
            labels_stat[uint_label] += 1

            used_row_ids.add(row[const.ID])

    return labels_stat
