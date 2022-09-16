from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.opinions.annot.algo.pair_based import PairBasedOpinionAnnotationAlgorithm
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms.base import SynonymsCollection
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.contrib.utils.pipelines.text_opinion.annot.algo_based import AlgorithmBasedTextOpinionAnnotator
from arekit.contrib.utils.pipelines.text_opinion.annot.predefined import PredefinedTextOpinionAnnotator
from arekit.contrib.utils.pipelines.text_opinion.filters.distance_based import DistanceLimitedTextOpinionFilter
from arekit.contrib.utils.pipelines.text_opinion.filters.entity_based import EntityBasedTextOpinionFilter
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection

from SentiNEREL.pipelines.etalon import create_etalon_pipeline, create_etalon_with_no_label_pipeline
from SentiNEREL.pipelines.test import create_test_pipeline
from SentiNEREL.pipelines.train import create_train_pipeline
from SentiNEREL.entity.filter import CollectionEntityFilter


def prepare_data_pipelines(text_parser, doc_ops, label_formatter, terms_per_context):
    """ Создаем словарь из pipelines для каждого типа данных.
    """

    train_neut_annot = create_nolabel_text_opinion_annotator(terms_per_context)
    test_neut_annot = create_nolabel_text_opinion_annotator(terms_per_context)

    text_opinion_filters = [
        EntityBasedTextOpinionFilter(entity_filter=CollectionEntityFilter()),
        DistanceLimitedTextOpinionFilter(terms_per_context)
    ]

    predefined_annot = PredefinedTextOpinionAnnotator(doc_ops, label_formatter)

    return {
        DataType.Train: create_train_pipeline(text_parser=text_parser,
                                              doc_ops=doc_ops,
                                              annotators=[
                                                  predefined_annot,
                                                  train_neut_annot
                                              ],
                                              text_opinion_filters=text_opinion_filters),
        DataType.Test: create_test_pipeline(text_parser=text_parser,
                                            doc_ops=doc_ops,
                                            annotators=[
                                                test_neut_annot
                                            ],
                                            text_opinion_filters=text_opinion_filters),
        DataType.Etalon: create_etalon_pipeline(text_parser=text_parser,
                                                doc_ops=doc_ops,
                                                predefined_annot=predefined_annot,
                                                text_opinion_filters=text_opinion_filters),
        DataType.Dev: create_etalon_with_no_label_pipeline(text_parser=text_parser,
                                                           doc_ops=doc_ops,
                                                           annotators=[
                                                               predefined_annot,
                                                               train_neut_annot
                                                           ],
                                                           text_opinion_filters=text_opinion_filters)
    }


def create_nolabel_text_opinion_annotator(terms_per_context, dist_in_sents=0, synonyms=None):
    """ TODO. Embeding this into AREkit.
    """
    assert(isinstance(terms_per_context, int))
    assert(isinstance(synonyms, SynonymsCollection) or synonyms is None)
    assert(isinstance(dist_in_sents, int))

    if synonyms is None:
        synonyms = StemmerBasedSynonymCollection(iter_group_values_lists=[],
                                                 stemmer=MystemWrapper(),
                                                 is_read_only=False,
                                                 debug=False)

    return AlgorithmBasedTextOpinionAnnotator(
        value_to_group_id_func=lambda value:
        SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
            synonyms=synonyms, value=value),
        annot_algo=PairBasedOpinionAnnotationAlgorithm(
            dist_in_sents=dist_in_sents,
            dist_in_terms_bound=terms_per_context,
            label_provider=ConstantLabelProvider(NoLabel())),
        create_empty_collection_func=lambda: OpinionCollection(
            opinions=[],
            synonyms=synonyms,
            error_on_duplicates=True,
            error_on_synonym_end_missed=False),
        get_doc_existed_opinions_func=lambda _: None)
