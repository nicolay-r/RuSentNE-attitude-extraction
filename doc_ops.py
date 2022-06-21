from arekit.common.experiment.api.ops_doc import DocumentOperations

from collection.reader import CollectionNewsReader


class CustomDocOperations(DocumentOperations):

    def __init__(self, exp_ctx, label_formatter, doc_ids):
        assert(isinstance(doc_ids, list))
        super(CustomDocOperations, self).__init__(exp_ctx)
        self.__label_formatter = label_formatter
        self.__doc_ids = doc_ids

    def get_doc(self, doc_id):
        return CollectionNewsReader.read_document(doc_id=doc_id,
                                                  label_formatter=self.__label_formatter,
                                                  keep_any_opinion=False)
