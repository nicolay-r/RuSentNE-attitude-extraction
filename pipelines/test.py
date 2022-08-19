from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.text.parser import BaseTextParser
from entity.filter import CollectionEntityFilter
from pipelines.train import text_opinion_extraction_pipeline


def create_test_pipeline(text_parser, doc_ops, annotators, terms_per_context):
    """ This is a pipeline for TEST data annotation.
        We perform annotation of the attitudes.
    """
    assert(isinstance(text_parser, BaseTextParser))
    assert(isinstance(annotators, list))
    assert(isinstance(doc_ops, DocumentOperations))
    assert(isinstance(terms_per_context, int))

    return text_opinion_extraction_pipeline(
        annotators=annotators,
        text_parser=text_parser,
        get_doc_func=lambda doc_id: doc_ops.get_doc(doc_id),
        terms_per_context=terms_per_context,
        entity_filter=CollectionEntityFilter())
