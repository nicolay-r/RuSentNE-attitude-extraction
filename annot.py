from arekit.common.entities.base import Entity
from arekit.common.entities.types import OpinionEntityType
from arekit.common.opinions.annot.algo.pair_based import PairBasedAnnotationAlgorithm

from entity.helper import EntityHelper


def is_entity_ignored(entity, e_type):
    """ субъектом всегда может быть только person, organization, country, profession
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


def create_annot(dist_in_terms_bound, label_provider):
    return PairBasedAnnotationAlgorithm(label_provider=label_provider,
                                        dist_in_terms_bound=dist_in_terms_bound,
                                        dist_in_sents=0,
                                        is_entity_ignored_func=is_entity_ignored)
