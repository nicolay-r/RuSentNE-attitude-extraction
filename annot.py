from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.news.parsed.providers.text_opinion_pairs import TextOpinionPairsProvider
from arekit.common.news.parsed.service import ParsedNewsService
from arekit.common.opinions.annot.algo.pair_based import PairBasedOpinionAnnotationAlgorithm
from arekit.common.opinions.annot.algo_based import AlgorithmBasedOpinionAnnotator
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection

from entity.filter import CollectionEntityFilter


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


class AlgorithmBasedTextOpinionAnnotator(AlgorithmBasedOpinionAnnotator):
    """ Обертка на OpinionAnnotator в которой выполняется конверсия в TextOpinions
    """

    def __init__(self, value_to_group_id_func, annot_algo, get_doc_existed_opinions_func, create_empty_collection_func):
        assert(callable(value_to_group_id_func))
        super(AlgorithmBasedTextOpinionAnnotator, self).__init__(
            annot_algo=annot_algo,
            create_empty_collection_func=create_empty_collection_func,
            get_doc_existed_opinions_func=get_doc_existed_opinions_func)
        self.__value_to_group_id_func = value_to_group_id_func

    def __create_service(self, parsed_news):
        return ParsedNewsService(parsed_news=parsed_news, providers=[
            TextOpinionPairsProvider(self.__value_to_group_id_func)
        ])

    def annotate_collection(self, parsed_news):
        service = self.__create_service(parsed_news)
        topp = service.get_provider(TextOpinionPairsProvider.NAME)
        for opinion in super(AlgorithmBasedTextOpinionAnnotator, self).annotate_collection(parsed_news):
            for text_opinion in topp.iter_from_opinion(opinion):
                yield text_opinion
