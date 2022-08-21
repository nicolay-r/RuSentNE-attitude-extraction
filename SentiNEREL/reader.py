from arekit.contrib.source.brat.annot import BratAnnotationParser
from arekit.contrib.source.brat.news import BratNews
from arekit.contrib.source.brat.sentences_reader import BratDocumentSentencesReader

from SentiNEREL.entities import CollectionEntityCollection
from SentiNEREL.io_utils import CollectionIOUtils, CollectionVersions


class SentiNERELDocReader(object):

    @staticmethod
    def read_text_relations(filename, doc_id, version):
        assert(isinstance(filename, str))
        assert(isinstance(doc_id, int))

        return CollectionIOUtils.read_from_zip(
            inner_path=CollectionIOUtils.get_annotation_innerpath(filename),
            process_func=lambda input_file: [
                relation for relation in BratAnnotationParser.parse_annotations(
                    input_file=input_file, encoding='utf-8-sig')["relations"]],
            version=version)

    @staticmethod
    def read_document(filename, doc_id):
        assert(isinstance(filename, str))
        assert(isinstance(doc_id, int))

        def file_to_doc(input_file):
            sentences = BratDocumentSentencesReader.from_file(input_file=input_file, entities=entities)
            return BratNews(doc_id=doc_id, sentences=sentences, text_relations=text_relations)

        entities = CollectionEntityCollection.read_collection(filename=filename, version=CollectionVersions.NO)
        text_relations = SentiNERELDocReader.read_text_relations(
            doc_id=doc_id, filename=filename, version=CollectionVersions.NO)

        return CollectionIOUtils.read_from_zip(
            inner_path=CollectionIOUtils.get_news_innerpath(filename=filename),
            process_func=file_to_doc,
            version=CollectionVersions.NO)
