from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.opinions.annot.algo.pair_based import PairBasedOpinionAnnotationAlgorithm
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.contrib.utils.pipelines.text_opinion.annot.algo_based import AlgorithmBasedTextOpinionAnnotator
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection


def create_neutral_annotator(terms_per_context):
    """ Default annotator, based on:
            - expandable synonyms collection
            - single label annotator.
    """

    synonyms = StemmerBasedSynonymCollection(iter_group_values_lists=[],
                                             stemmer=MystemWrapper(),
                                             is_read_only=False,
                                             debug=False)

    return AlgorithmBasedTextOpinionAnnotator(
        value_to_group_id_func=lambda value:
            SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                synonyms=synonyms, value=value),
        annot_algo=PairBasedOpinionAnnotationAlgorithm(
            dist_in_sents=0,
            dist_in_terms_bound=terms_per_context,
            label_provider=ConstantLabelProvider(NoLabel())),
        create_empty_collection_func=lambda: OpinionCollection(
            opinions=[],
            synonyms=synonyms,
            error_on_duplicates=True,
            error_on_synonym_end_missed=False),
        get_doc_existed_opinions_func=lambda _: None)
