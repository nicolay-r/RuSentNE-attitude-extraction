from arekit.common.experiment.api.ops_doc import DocumentOperations

from collection.reader import CollectionNewsReader


class CustomDocOperations(DocumentOperations):

    def __init__(self, label_formatter, filename_by_id):
        """ filename_ids: dict
                Dictionary of {id: filename}, where
                    - id: int
                    - filename: str
        """
        assert(isinstance(filename_by_id, dict))
        super(CustomDocOperations, self).__init__()
        self.__label_formatter = label_formatter
        self.__filename_by_id = filename_by_id

    def get_doc(self, doc_id):
        return CollectionNewsReader.read_document(doc_id=doc_id,
                                                  filename=self.__filename_by_id[doc_id],
                                                  label_formatter=self.__label_formatter,
                                                  keep_any_opinion=False)
