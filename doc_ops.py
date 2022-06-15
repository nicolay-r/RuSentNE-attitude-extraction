from arekit.common.experiment.api.ops_doc import DocumentOperations


class CustomDocOperations(DocumentOperations):

    def __init__(self, exp_ctx):
        super(CustomDocOperations, self).__init__(exp_ctx)
        self.__docs = None
        self.__doc_ids = None

    def set_docs(self, docs):
        assert(isinstance(docs, list))
        self.__docs = docs
        self.__doc_ids = list(range(len(self.__docs)))

    def get_doc(self, doc_id):
        return self.__docs[doc_id]