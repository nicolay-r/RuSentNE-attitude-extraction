from arekit.common.experiment.data_type import DataType
from arekit.common.folding.fixed import FixedFolding

from folding.utils import create_filenames_by_ids


def create_train_test_folding(train_filenames, test_filenames):
    """ Create fixed datafolding based on the predefined list of filenames,
        written in file.
    """
    assert(isinstance(train_filenames, list))
    assert(isinstance(test_filenames, list))

    filenames_by_ids = create_filenames_by_ids(filenames=train_filenames + test_filenames)

    ids_by_filenames = {}
    for doc_id, filename in filenames_by_ids.items():
        ids_by_filenames[filename] = doc_id

    train_doc_ids = [ids_by_filenames[filename] for filename in train_filenames]
    test_doc_ids = [ids_by_filenames[filename] for filename in test_filenames]

    folding = FixedFolding.from_parts({DataType.Train: train_doc_ids, DataType.Test: test_doc_ids})

    return filenames_by_ids, folding

