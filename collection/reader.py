from arekit.common.entities.collection import EntityCollection
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.source.brat.annot import BratAnnotationParser
from arekit.contrib.source.brat.news_reader import BratDocumentSentencesReader
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection

from collection.entities import CollectionEntityCollection
from collection.io_utils import CollectionIOUtils, CollectionVersions
from collection.news import CustomNews
from collection.opinions.converter import CollectionOpinionConverter


class CollectionNewsReader(object):

    @staticmethod
    def read_text_opinions(filename, doc_id, entities, version, label_formatter, keep_any_type):
        assert(isinstance(filename, str))
        assert(isinstance(label_formatter, StringLabelsFormatter))
        assert(isinstance(entities, EntityCollection))
        assert(isinstance(doc_id, int))

        return CollectionIOUtils.read_from_zip(
            inner_path=CollectionIOUtils.get_annotation_innerpath(filename),
            process_func=lambda input_file: [
                CollectionOpinionConverter.to_text_opinion(relation, doc_id=doc_id, label_formatter=label_formatter)
                for relation in
                BratAnnotationParser.parse_annotations(input_file=input_file, encoding='utf-8-sig')["relations"]
                if label_formatter.supports_value(relation.Type) or keep_any_type],
            version=version)

    @staticmethod
    def read_document(filename, doc_id, label_formatter, keep_any_opinion):
        assert(isinstance(filename, str))
        assert(isinstance(doc_id, int))
        assert(isinstance(label_formatter, StringLabelsFormatter))
        assert(isinstance(keep_any_opinion, bool))

        def file_to_doc(input_file):
            sentences = BratDocumentSentencesReader.from_file(input_file=input_file, entities=entities)
            return CustomNews(doc_id=doc_id,
                              sentences=sentences,
                              text_opinions=opinions)

        synonyms = StemmerBasedSynonymCollection(iter_group_values_lists=[],
                                                 stemmer=MystemWrapper(),
                                                 is_read_only=False,
                                                 debug=False)

        entities = CollectionEntityCollection.read_collection(
            filename=filename, synonyms=synonyms, version=CollectionVersions.NO)

        opinions = CollectionNewsReader.read_text_opinions(
            doc_id=doc_id,
            filename=filename,
            entities=entities,
            version=CollectionVersions.NO,
            label_formatter=label_formatter,
            keep_any_type=keep_any_opinion)

        return CollectionIOUtils.read_from_zip(
            inner_path=CollectionIOUtils.get_news_innerpath(filename=filename),
            process_func=file_to_doc,
            version=CollectionVersions.NO)
