from arekit.common.entities.base import Entity
from arekit.common.entities.types import OpinionEntityType
from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.opinions.annot.algo.pair_based import PairBasedOpinionAnnotationAlgorithm
from arekit.common.opinions.annot.algo_based import AlgorithmBasedOpinionAnnotator
from arekit.common.opinions.collection import OpinionCollection
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection
from arekit.processing.lemmatization.mystem import MystemWrapper

from entity.helper import EntityHelper


def is_entity_ignored(entity, e_type):
    """ субъектом всегда может быть только:
            [PERSON, ORGANIZATION, COUNTRY, PROFESSION]
        — объектом могут быть видимо все типы.
    """
    assert(isinstance(entity, Entity))
    assert(isinstance(e_type, OpinionEntityType))

    subjects = [EntityHelper.PERSON, EntityHelper.ORGANIZATION, EntityHelper.COUNTRY, EntityHelper.PROFESSION]
    objects_ignored = [EntityHelper.AGE, EntityHelper.NUMBER, EntityHelper.PERCENT, EntityHelper.ORDINAL,
                       EntityHelper.TIME, EntityHelper.MONEY, EntityHelper.DATE]

    if e_type == OpinionEntityType.Subject:
        return entity.Type not in subjects
    if e_type == OpinionEntityType.Object:
        return entity.Type in objects_ignored
    else:
        return True


def create_neutral_annotator(terms_per_context):
    """ Default annotator, based on:
            - expandable synonyms collection
            - single label annotator.
    """

    synonyms = StemmerBasedSynonymCollection(iter_group_values_lists=[],
                                             stemmer=MystemWrapper(),
                                             is_read_only=False,
                                             debug=False)

    annotator = AlgorithmBasedOpinionAnnotator(
        annot_algo=PairBasedOpinionAnnotationAlgorithm(
            dist_in_sents=0,
            is_entity_ignored_func=is_entity_ignored,
            dist_in_terms_bound=terms_per_context,
            label_provider=ConstantLabelProvider(NoLabel())),
        create_empty_collection_func=lambda: OpinionCollection(
            opinions=[],
            synonyms=synonyms,
            error_on_duplicates=True,
            error_on_synonym_end_missed=False),
        get_doc_existed_opinions_func=lambda _: None)

    return annotator, synonyms