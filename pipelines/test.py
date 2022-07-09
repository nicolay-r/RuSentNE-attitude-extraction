from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.opinions.annot.base import BaseOpinionAnnotator
from arekit.common.synonyms.base import SynonymsCollection
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.utils.pipelines.annot.base import attitude_extraction_default_pipeline


def create_test_pipeline(text_parser, doc_ops, test_annotator, synonyms, terms_per_context):
    """ This is a pipeline for TEST data annotation.
        We perform annotation of the attitudes.
    """
    assert(isinstance(text_parser, BaseTextParser))
    assert(isinstance(test_annotator, BaseOpinionAnnotator))
    assert(isinstance(doc_ops, DocumentOperations))
    assert(isinstance(synonyms, SynonymsCollection))
    assert(isinstance(terms_per_context, int))

    return attitude_extraction_default_pipeline(
        annotator=test_annotator,
        get_doc_func=lambda doc_id: doc_ops.get_doc(doc_id),
        text_parser=text_parser,
        value_to_group_id_func=lambda value:
            SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                synonyms=synonyms, value=value),
        terms_per_context=terms_per_context,
        entity_index_func=lambda brat_entity: brat_entity.ID)
