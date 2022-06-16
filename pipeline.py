from arekit.common.data.input.sample import InputSampleBase
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.news.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.news.parsed.service import ParsedNewsService
from arekit.common.news.parser import NewsParser
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.item_map import MapPipelineItem
from arekit.common.pipeline.items.flatten import FlattenIterPipelineItem
from arekit.common.text_opinions.base import TextOpinion
from collection.reader import CollectionNewsReader


def __to_text_opinion_linkages(parsed_news, text_opinions, filter_func, tag_value_func):
    assert(isinstance(parsed_news, ParsedNews))
    assert(callable(filter_func))

    for text_opinion in text_opinions:
        assert(isinstance(text_opinion, TextOpinion))

        if parsed_news.find_entity(text_opinion.SourceId) is None or \
           parsed_news.find_entity(text_opinion.TargetId) is None:
            continue

        linkage = TextOpinionsLinkage(
            [TextOpinion.create_copy(text_opinion, keep_text_opinion_id=False)]
        )

        if tag_value_func is not None:
            linkage.set_tag(tag_value_func(linkage))

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
            ParsedNewsService(parsed_news=data[1], providers=[EntityServiceProvider()]),
            data[1])),

        # (news, service, parsed_news) -> (text_opinions).
        MapPipelineItem(map_func=lambda data: __to_text_opinion_linkages(
            parsed_news=data[2],
            tag_value_func=lambda _: data[1],
            text_opinions=data[0].TextOpinions,
            filter_func=lambda text_opinion: InputSampleBase.check_ability_to_create_sample(
                entity_service=data[2].get_provider(EntityServiceProvider.NAME),
                text_opinion=text_opinion,
                window_size=terms_per_context))),

        # linkages[] -> linkages.
        FlattenIterPipelineItem()
    ])
