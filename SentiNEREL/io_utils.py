from enum import Enum
from os import path
from os.path import basename, dirname, realpath, join

from arekit.contrib.source.zip_utils import ZipArchiveUtils


class CollectionVersions(Enum):
    """ List of the supported version of this collection
    """

    NO = "None"


class CollectionIOUtils(ZipArchiveUtils):

    # Represents an root archive.
    inner_root = "sentiment_dataset"

    @staticmethod
    def get_archive_filepath(version):
        # local_path = CollectionIOUtils.get_data_root()
        current_dir = dirname(realpath(__file__))
        local_path = join(current_dir, "../data")
        return path.join(local_path, "sentiment_dataset.zip".format(version))

    # region internal methods

    @staticmethod
    def get_annotation_innerpath(filename):
        assert(isinstance(filename, str))
        return path.join(CollectionIOUtils.inner_root, "{}.ann".format(filename))

    @staticmethod
    def get_news_innerpath(filename):
        assert(isinstance(filename, str))
        return path.join(CollectionIOUtils.inner_root, "{}.txt".format(filename))

    # endregion

    @staticmethod
    def __iter_filenames_from_dataset(folder_name):
        assert(isinstance(folder_name, str))

        for filename in CollectionIOUtils.iter_filenames_from_zip(CollectionVersions.NO):

            extension = filename[-4:]

            # Crop extension.
            filename = filename[:-4]

            if extension != ".txt":
                continue

            if not folder_name in filename:
                continue

            yield basename(filename)

    # region public methods

    @staticmethod
    def iter_collection_filenames():
        filenames_it = CollectionIOUtils.__iter_filenames_from_dataset(
            folder_name=CollectionIOUtils.inner_root)

        for doc_id, filename in enumerate(filenames_it):
            yield doc_id, filename

    # endregion
