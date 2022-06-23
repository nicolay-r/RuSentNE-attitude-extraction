from arekit.common.data.input.sample import InputSampleBase
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.news.parsed.providers.base import BaseParsedNewsServiceProvider
from arekit.common.news.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.news.parsed.service import ParsedNewsService
from arekit.common.news.parsed.term_position import TermPositionTypes
from arekit.common.news.parser import NewsParser
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.item_map import MapPipelineItem
from arekit.common.pipeline.items.flatten import FlattenIterPipelineItem
from arekit.common.text_opinions.base import TextOpinion

from collection.news import CustomNews


def create_train_pipeline(text_parser, doc_ops, terms_per_context):
    """ Train pipeline is based on the predefined annotations.
    """
    return text_opinions_to_opinion_linkages_pipeline(
        terms_per_context=terms_per_context,
        get_doc_func=lambda doc_id: doc_ops.get_doc(doc_id),
        text_parser=text_parser)


def __convert_opinion_id(news, origin_id, esp):
    assert(isinstance(news, CustomNews))
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


def __to_text_opinion_linkages(news, parsed_news, filter_func, parsed_news_service):
    assert(isinstance(news, CustomNews))
    assert(isinstance(parsed_news, ParsedNews))
    assert(callable(filter_func))
    assert(isinstance(parsed_news_service, ParsedNewsService))

    esp = parsed_news_service.get_provider("entity-service-provider")

    for text_opinion in news.TextOpinions:
        assert(isinstance(text_opinion, TextOpinion))

        internal_opinion = text_opinion.try_convert(
            other=text_opinion,
            convert_func=lambda origin_id: __convert_opinion_id(news=news, origin_id=origin_id, esp=esp))

        if internal_opinion is None:
            continue

        s_ind = esp.get_entity_position(internal_opinion.SourceId, position_type=TermPositionTypes.SentenceIndex)
        t_ind = esp.get_entity_position(internal_opinion.TargetId, position_type=TermPositionTypes.SentenceIndex)

        if s_ind != t_ind:
            # AREkit does not provide a support for multi-sentence opinions at present.
            continue

        linkage = TextOpinionsLinkage([internal_opinion])
        linkage.set_tag(parsed_news_service)
        yield linkage


def text_opinions_to_opinion_linkages_pipeline(text_parser, get_doc_func, terms_per_context):
    assert(callable(get_doc_func))
    assert(isinstance(terms_per_context, int))

    return BasePipeline([
        # (doc_id) -> (news)
        MapPipelineItem(map_func=lambda doc_id: get_doc_func(doc_id)),

        # (news) -> (news, parsed_news)
        MapPipelineItem(map_func=lambda news: (news, NewsParser.parse(news, text_parser))),

        # (news, parsed_news) -> (news, service, parsed_news)
        MapPipelineItem(map_func=lambda data: (
            data[0],
            ParsedNewsService(parsed_news=data[1], providers=[
                EntityServiceProvider(lambda brat_entity: brat_entity.ID)
            ]),
            data[1])),

        # (news, service, parsed_news) -> (text_opinions).
        MapPipelineItem(map_func=lambda data: __to_text_opinion_linkages(
            news=data[0],
            parsed_news=data[2],
            parsed_news_service=data[1],
            filter_func=lambda text_opinion: InputSampleBase.check_ability_to_create_sample(
                entity_service=data[2].get_provider(EntityServiceProvider.NAME),
                text_opinion=text_opinion,
                window_size=terms_per_context))),

        # linkages[] -> linkages.
        FlattenIterPipelineItem()
    ])
