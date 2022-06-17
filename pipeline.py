from arekit.common.data.input.sample import InputSampleBase
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.news.parsed.providers.base import BaseParsedNewsServiceProvider
from arekit.common.news.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.news.parsed.service import ParsedNewsService
from arekit.common.news.parser import NewsParser
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.item_map import MapPipelineItem
from arekit.common.pipeline.items.flatten import FlattenIterPipelineItem
from arekit.common.text_opinions.base import TextOpinion

from collection.news import CustomNews
from collection.reader import CollectionNewsReader


def __convert_opinion_to_internal(news, text_opinion, esp):
    assert(isinstance(esp, BaseParsedNewsServiceProvider))

    if not news.contains_entity(text_opinion.SourceId) or not news.contains_entity(text_opinion.TargetId):
        # Due to the complexity of entities, some entities might be nested.
        # Therefore the latter, some entities might be discarded.
        return None

    entity_source = news.get_entity_by_id(text_opinion.SourceId)
    entity_target = news.get_entity_by_id(text_opinion.TargetId)

    if not esp.contains_entity(entity_source) or not esp.contains_entity(entity_target):
        return None

    document_entity_source = esp.get_document_entity(entity_source)
    document_entity_target = esp.get_document_entity(entity_source)

    return TextOpinion(doc_id=news.ID,
                       text_opinion_id=None,
                       source_id=document_entity_source.IdInDocument,
                       target_id=document_entity_target.IdInDocument,
                       owner=text_opinion.Owner,
                       label=text_opinion.Sentiment)


def __to_text_opinion_linkages(news, parsed_news, filter_func, parsed_news_service):
    assert(isinstance(news, CustomNews))
    assert(isinstance(parsed_news, ParsedNews))
    assert(callable(filter_func))
    assert(isinstance(parsed_news_service, ParsedNewsService))

    esp = parsed_news_service.get_provider("entity-service-provider")

    for text_opinion in news.TextOpinions:
        assert(isinstance(text_opinion, TextOpinion))

        internal_opinion = __convert_opinion_to_internal(news=news, text_opinion=text_opinion, esp=esp)

        if internal_opinion is None:
            continue

        linkage = TextOpinionsLinkage([internal_opinion])
        linkage.set_tag(parsed_news_service)
        yield linkage


def text_opinions_to_opinion_linkages_pipeline(text_parser, label_formatter, terms_per_context):
    assert(isinstance(terms_per_context, int))

    return BasePipeline([
        # (doc_id) -> (news)
        MapPipelineItem(map_func=lambda doc_id:
        CollectionNewsReader.read_document(doc_id=doc_id, label_formatter=label_formatter)),

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
