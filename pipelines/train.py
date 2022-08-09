from arekit.common.data.input.sample import InputSampleBase
from arekit.common.entities.types import OpinionEntityType
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.news.parsed.providers.base import BaseParsedNewsServiceProvider
from arekit.common.news.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.news.parsed.providers.text_opinion_pairs import TextOpinionPairsProvider
from arekit.common.news.parsed.service import ParsedNewsService
from arekit.common.news.parsed.term_position import TermPositionTypes
from arekit.common.news.parser import NewsParser
from arekit.common.opinions.annot.base import BaseOpinionAnnotator
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.item_map import MapPipelineItem
from arekit.common.pipeline.items.flatten import FlattenIterPipelineItem
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text_opinions.base import TextOpinion
from arekit.contrib.source.brat.news import BratNews

from entity.filter import EntityFilter, CollectionEntityFilter


def create_train_pipeline(text_parser, doc_ops, annotator, synonyms, terms_per_context):
    """ Train pipeline is based on the predefined annotations and
        automatic annotations of other pairs with a NoLabel.

        # TODO. Provide here the list of annotators.
    """
    assert(isinstance(annotator, BaseOpinionAnnotator) or annotator is None)

    return text_opinions_to_opinion_linkages_pipeline(
        terms_per_context=terms_per_context,
        get_doc_func=lambda doc_id: doc_ops.get_doc(doc_id),
        text_parser=text_parser,
        neut_annotator=annotator,
        entity_filter=CollectionEntityFilter(),
        value_to_group_id_func=lambda value:
            SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                synonyms=synonyms, value=value))


def __convert_opinion_id(news, origin_id, esp):
    assert(isinstance(news, BratNews))
    assert(isinstance(origin_id, int))
    assert(isinstance(esp, BaseParsedNewsServiceProvider))

    if not news.contains_entity(origin_id):
        # Due to the complexity of entities, some entities might be nested.
        # Therefore the latter, some entities might be discarded.
        return None

    origin_entity = news.get_entity_by_id(origin_id)

    if not esp.contains_entity(origin_entity):
        return None

    document_entity = esp.get_document_entity(origin_entity)
    return document_entity.IdInDocument


def __filter_internal_opinion(internal_opinion, esp, terms_per_context, entity_filter):
    """ This method describes internal opinion filtering
    """
    assert(isinstance(entity_filter, EntityFilter) or entity_filter is None)
    assert(isinstance(internal_opinion, TextOpinion) or internal_opinion is None)

    if internal_opinion is None:
        return False

    s_ind = esp.get_entity_position(internal_opinion.SourceId,
                                    position_type=TermPositionTypes.SentenceIndex)
    t_ind = esp.get_entity_position(internal_opinion.TargetId,
                                    position_type=TermPositionTypes.SentenceIndex)

    if s_ind != t_ind:
        # AREkit does not provide a support for multi-sentence opinions at present.
        return False

    # TODO. Move this away into the related annotator.
    # TODO. Annotator, based on the predefined annotations in the original collection.
    e_source = esp._doc_entities[internal_opinion.SourceId]
    if entity_filter is not None and entity_filter.is_ignored(e_source, OpinionEntityType.Subject):
        return False

    # TODO. Move this away into the related annotator.
    # TODO. Annotator, based on the predefined annotations in the original collection.
    e_target = esp._doc_entities[internal_opinion.TargetId]
    if entity_filter is not None and entity_filter.is_ignored(e_target, OpinionEntityType.Object):
        return False

    return InputSampleBase.check_ability_to_create_sample(
        entity_service=esp,
        text_opinion=internal_opinion,
        window_size=terms_per_context)


def iter_train_text_opinion_linkages(news, parsed_news, annotator, parsed_news_service,
                                     terms_per_context, entity_filter):
    """ #TODO. Provide a couple of annotators.
    """
    assert(isinstance(news, BratNews))
    assert(isinstance(annotator, BaseOpinionAnnotator) or annotator is None)
    assert(isinstance(parsed_news, ParsedNews))
    assert(isinstance(parsed_news_service, ParsedNewsService))
    assert(isinstance(terms_per_context, int))
    assert(isinstance(entity_filter, EntityFilter))

    def __to_id(text_opinion):
        return "{}_{}".format(text_opinion.SourceId, text_opinion.TargetId)

    esp = parsed_news_service.get_provider(EntityServiceProvider.NAME)
    topp = parsed_news_service.get_provider(TextOpinionPairsProvider.NAME)

    predefined = set()

    # TODO. Provide a couple of annotators.
    # TODO. This is the first annotator.
    # TODO. It would be nice to see the loop over annotators.
    # Predefined sentiment annotation.
    for text_opinion in news.TextOpinions:
        assert(isinstance(text_opinion, TextOpinion))

        internal_opinion = text_opinion.try_convert(
            other=text_opinion,
            convert_func=lambda origin_id: __convert_opinion_id(news=news, origin_id=origin_id, esp=esp))

        keep_internal_opinion = __filter_internal_opinion(internal_opinion=internal_opinion,
                                                          esp=esp,
                                                          terms_per_context=terms_per_context,
                                                          entity_filter=entity_filter)

        if not keep_internal_opinion:
            continue

        predefined.add(__to_id(internal_opinion))

        linkage = TextOpinionsLinkage([internal_opinion])
        linkage.set_tag(parsed_news_service)
        yield linkage

    # TODO. This is the second annotator.
    # TODO. It would be nice to see the loop over annotators.
    # Neutral annotation.
    # Выполнено через следующий механизм: сначала выполняется разметка на уровне
    # отношений документа (annotator), а потом из них выполняется преобразование в контекстыные отношения
    # с выполнением проверки на корректность, а также отбросом отношений которые были в размете.

    if annotator is None:
        return

    for opinion in annotator.annotate_collection(parsed_news=parsed_news):
        for neut_text_opinion in topp.iter_from_opinion(opinion):
            assert(isinstance(neut_text_opinion, TextOpinion))

            keep_internal_opinion = __filter_internal_opinion(internal_opinion=neut_text_opinion,
                                                              esp=esp,
                                                              terms_per_context=terms_per_context,
                                                              entity_filter=entity_filter)

            if not keep_internal_opinion:
                continue

            if __to_id(neut_text_opinion) in predefined:
                # We reject those one which was already obtained from the predefined sentiment annotation.
                continue

            linkage = TextOpinionsLinkage([neut_text_opinion])
            linkage.set_tag(parsed_news_service)
            yield linkage


def text_opinions_to_opinion_linkages_pipeline(text_parser, get_doc_func, neut_annotator, terms_per_context,
                                               value_to_group_id_func, entity_filter):
    assert(callable(get_doc_func))
    assert(isinstance(terms_per_context, int))
    assert(callable(value_to_group_id_func))
    assert(isinstance(entity_filter, EntityFilter))

    return BasePipeline([
        # (doc_id) -> (news)
        MapPipelineItem(map_func=lambda doc_id: get_doc_func(doc_id)),

        # (news) -> (news, parsed_news)
        MapPipelineItem(map_func=lambda news: (news, NewsParser.parse(news, text_parser))),

        # (news, parsed_news) -> (news, service, parsed_news)
        MapPipelineItem(map_func=lambda data: (
            data[0],
            ParsedNewsService(parsed_news=data[1], providers=[
                EntityServiceProvider(lambda brat_entity: brat_entity.ID),
                TextOpinionPairsProvider(value_to_group_id_func),
            ]),
            data[1])),

        # (news, service, parsed_news) -> (text_opinions).
        MapPipelineItem(map_func=lambda data: iter_train_text_opinion_linkages(
            news=data[0],
            annotator=neut_annotator,
            parsed_news=data[2],
            parsed_news_service=data[1],
            terms_per_context=terms_per_context,
            entity_filter=entity_filter)),

        # linkages[] -> linkages.
        FlattenIterPipelineItem()
    ])
