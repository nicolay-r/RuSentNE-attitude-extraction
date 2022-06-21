from arekit.common.entities.collection import EntityCollection
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.source.brat.annot import BratAnnotationParser
from arekit.contrib.source.brat.news_reader import BratDocumentSentencesReader
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection
from arekit.processing.lemmatization.mystem import MystemWrapper

from collection.entities import CollectionEntityCollection
from collection.io_utils import CollectionIOUtils, CollectionVersions
from collection.news import CustomNews
from collection.opinions.converter import CollectionOpinionConverter


class CollectionNewsReader(object):

    @staticmethod
    def read_opinions(doc_id, entities, version, label_formatter):
        assert(isinstance(label_formatter, StringLabelsFormatter))
        assert(isinstance(entities, EntityCollection))
        assert(isinstance(doc_id, int))

        return CollectionIOUtils.read_from_zip(
            inner_path=CollectionIOUtils.get_annotation_innerpath(doc_id),
            process_func=lambda input_file: [
                CollectionOpinionConverter.to_text_opinion(relation, doc_id=doc_id, label_formatter=label_formatter)
                for relation in
                BratAnnotationParser.parse_annotations(input_file=input_file, encoding='utf-8-sig')["relations"]
                if label_formatter.supports_value(relation.Type)],
            version=version)

    @staticmethod
    def read_document(doc_id, label_formatter):
        assert(isinstance(label_formatter, StringLabelsFormatter))

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
            doc_id=doc_id, synonyms=synonyms, version=CollectionVersions.NO)

        opinions = CollectionNewsReader.read_opinions(
            doc_id=doc_id, entities=entities, version=CollectionVersions.NO, label_formatter=label_formatter)

        return CollectionIOUtils.read_from_zip(
            inner_path=CollectionIOUtils.get_news_innerpath(doc_id),
            process_func=file_to_doc,
            version=CollectionVersions.NO)
