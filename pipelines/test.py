from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.opinions.annot.base import BaseAnnotator
from arekit.common.synonyms import SynonymsCollection
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.utils.pipelines.annot.base import attitude_extraction_default_pipeline
from collection.entities import CollectionEntityCollection


def create_test_pipeline(text_parser, doc_ops, neut_annotator, synonyms, terms_per_context):
    """ This is a pipeline for TEST data annotation.
        We perform annotation of the attitudes.
    """
    assert(isinstance(text_parser, BaseTextParser))
    assert(isinstance(neut_annotator, BaseAnnotator))
    assert(isinstance(doc_ops, DocumentOperations))
    assert(isinstance(synonyms, SynonymsCollection))
    assert(isinstance(terms_per_context, int))

    return attitude_extraction_default_pipeline(
        annotator=neut_annotator,
        data_type=DataType.Test,
        get_doc_func=lambda doc_id: doc_ops.get_doc(doc_id),
        text_parser=text_parser,
        value_to_group_id_func=lambda value: CollectionEntityCollection.get_synonym_group_index_or_add(
            synonyms=synonyms, value=value),
        terms_per_context=terms_per_context,
        entity_index_func=lambda brat_entity: brat_entity.ID)
