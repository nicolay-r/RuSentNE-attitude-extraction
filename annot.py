from arekit.common.opinions.annot.base import BaseAnnotator


class AnnotatorPlaceholder(BaseAnnotator):

    def annotate_collection(self, data_type, parsed_news):
        """ We have list of an already existed text opinions.
            There is a need to just compose the related collection.
        """
        return parsed_news.RelatedDocID, \
               self._annot_collection_core(parsed_news=parsed_news, data_type=data_type)
