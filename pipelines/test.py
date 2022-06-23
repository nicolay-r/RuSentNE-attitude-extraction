from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.opinions.annot.algo.pair_based import PairBasedAnnotationAlgorithm
from arekit.common.opinions.annot.default import DefaultAnnotator
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.utils.pipelines.annot.base import attitude_extraction_default_pipeline
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection
from arekit.processing.lemmatization.mystem import MystemWrapper

from collection.entities import CollectionEntityCollection


def create_test_pipeline(text_parser, doc_ops, terms_per_context):
    """ This is a pipeline for TEST data annotation.
        We perform annotation of the attitudes.
    """
    assert(isinstance(text_parser, BaseTextParser))
    assert(isinstance(doc_ops, DocumentOperations))
    assert(isinstance(terms_per_context, int))

    test_synonyms = StemmerBasedSynonymCollection(iter_group_values_lists=[],
                                                  stemmer=MystemWrapper(),
                                                  is_read_only=False,
                                                  debug=False)

    return attitude_extraction_default_pipeline(
        annotator=DefaultAnnotator(
            annot_algo=PairBasedAnnotationAlgorithm(
                dist_in_terms_bound=terms_per_context,
                label_provider=ConstantLabelProvider(NoLabel())),
            create_empty_collection_func=lambda: OpinionCollection(
                opinions=[],
                synonyms=test_synonyms,
                error_on_duplicates=True,
                error_on_synonym_end_missed=False),
            get_doc_etalon_opins_func=lambda _: []),
        data_type=DataType.Test,
        get_doc_func=lambda doc_id: doc_ops.get_doc(doc_id),
        text_parser=text_parser,
        value_to_group_id_func=lambda value: CollectionEntityCollection.get_synonym_group_index_or_add(
            synonyms=test_synonyms, value=value),
        terms_per_context=terms_per_context,
        entity_index_func=lambda brat_entity: brat_entity.ID)
