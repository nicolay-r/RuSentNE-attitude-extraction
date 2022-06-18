import os
from enum import Enum
from os import path

from arekit.contrib.source.zip_utils import ZipArchiveUtils


class CollectionVersions(Enum):
    NO = "None"


class CollectionIOUtils(ZipArchiveUtils):

    @staticmethod
    def get_archive_filepath(version):
        # local_path = CollectionIOUtils.get_data_root()
        local_path = "data"
        return path.join(local_path, "sentiment_dataset.zip".format(version))

    # region internal methods

    @staticmethod
    def get_annotation_innerpath(index):
        inner_root = "sentiment_dataset"
        return path.join(inner_root, "{}_text.ann".format(index))

    @staticmethod
    def get_news_innerpath(index):
        assert(isinstance(index, int))
        inner_root = "sentiment_dataset"
        return path.join(inner_root, "{}_text.txt".format(index))

    # endregion

    @staticmethod
    def __number_from_string(s):
        digit_chars_prefix = []

        for chr in s:
            if chr.isdigit():
                digit_chars_prefix.append(chr)
            else:
                break

        if len(digit_chars_prefix) == 0:
            return None

        return int("".join(digit_chars_prefix))

    @staticmethod
    def __iter_indicies_from_dataset(folder_name):
        assert(isinstance(folder_name, str))

        used = set()

        for filename in CollectionIOUtils.iter_filenames_from_zip(CollectionVersions.NO):
            if not folder_name in filename:
                continue

            index = CollectionIOUtils.__number_from_string(os.path.basename(filename))

            if index is None:
                continue

            if index in used:
                continue

            used.add(index)

            yield index

    # region public methods

    @staticmethod
    def iter_collection_indices():
        for index in CollectionIOUtils.__iter_indicies_from_dataset(folder_name="sentiment_dataset/"):
            yield index

    # endregion
