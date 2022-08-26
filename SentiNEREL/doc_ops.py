from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.contrib.source.sentinerel.reader import SentiNerelDocReader


class CollectionDocOperation(DocumentOperations):

    def __init__(self, filename_by_id):
        """ filename_ids: dict
                Dictionary of {id: filename}, where
                    - id: int
                    - filename: str
        """
        assert(isinstance(filename_by_id, dict))
        super(CollectionDocOperation, self).__init__()
        self.__filename_by_id = filename_by_id

    def get_doc(self, doc_id):
        return SentiNerelDocReader.read_document(doc_id=doc_id, filename=self.__filename_by_id[doc_id])
