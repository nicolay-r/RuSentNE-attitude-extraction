from arekit.common.data.input.sample import InputSampleBase
from arekit.common.entities.types import OpinionEntityType
from arekit.common.linkage.text_opinions import TextOpinionsLinkage
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.news.parsed.providers.entity_service import EntityServiceProvider
from arekit.common.news.parsed.service import ParsedNewsService
from arekit.common.news.parsed.term_position import TermPositionTypes
from arekit.common.news.parser import NewsParser
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.item_map import MapPipelineItem
from arekit.common.pipeline.items.flatten import FlattenIterPipelineItem
from arekit.common.text_opinions.base import TextOpinion

from entity.filter import EntityFilter, CollectionEntityFilter


def create_train_pipeline(text_parser, doc_ops, annotators, terms_per_context):
    """ Train pipeline is based on the predefined annotations and
        automatic annotations of other pairs with a NoLabel.
    """
    return text_opinion_extraction_pipeline(
        terms_per_context=terms_per_context,
        get_doc_func=lambda doc_id: doc_ops.get_doc(doc_id),
        text_parser=text_parser,
        annotators=annotators,
        entity_filter=CollectionEntityFilter())


def __filter_internal_opinion(internal_opinion, esp, terms_per_context, entity_filter):
    """ This method describes internal opinion filtering
        TODO. Make an external function for that.
        TODO. Make an external function for that.
        TODO. Make an external function for that.
    """
    assert(isinstance(entity_filter, EntityFilter) or entity_filter is None)
    assert(isinstance(internal_opinion, TextOpinion) or internal_opinion is None)

    if internal_opinion is None:
        return False

    s_ind = esp.get_entity_position(internal_opinion.SourceId, position_type=TermPositionTypes.SentenceIndex)
    t_ind = esp.get_entity_position(internal_opinion.TargetId, position_type=TermPositionTypes.SentenceIndex)

    if s_ind != t_ind:
        # AREkit does not provide a support for multi-sentence opinions at present.
        return False

    e_source = esp._doc_entities[internal_opinion.SourceId]
    if entity_filter is not None and entity_filter.is_ignored(e_source, OpinionEntityType.Subject):
        return False

    e_target = esp._doc_entities[internal_opinion.TargetId]
    if entity_filter is not None and entity_filter.is_ignored(e_target, OpinionEntityType.Object):
        return False

    return InputSampleBase.check_ability_to_create_sample(
        entity_service=esp,
        text_opinion=internal_opinion,
        window_size=terms_per_context)


def iter_text_opinion_linkages(parsed_news, annotators, terms_per_context, entity_filter):
    assert(isinstance(annotators, list))
    assert(isinstance(parsed_news, ParsedNews))
    assert(isinstance(terms_per_context, int))
    assert(isinstance(entity_filter, EntityFilter))

    def __to_id(text_opinion):
        return "{}_{}".format(text_opinion.SourceId, text_opinion.TargetId)

    service = ParsedNewsService(parsed_news=parsed_news, providers=[
        EntityServiceProvider(lambda brat_entity: brat_entity.ID),
    ])

    esp = service.get_provider(EntityServiceProvider.NAME)

    predefined = set()
    for annotator in annotators:
        for text_opinion in annotator.annotate_collection(parsed_news=parsed_news):
            assert(isinstance(text_opinion, TextOpinion))

            keep_internal_opinion = __filter_internal_opinion(
                internal_opinion=text_opinion, esp=esp,
                terms_per_context=terms_per_context, entity_filter=entity_filter)

            if not keep_internal_opinion:
                continue

            if __to_id(text_opinion) in predefined:
                # We reject those one which was already obtained
                # from the predefined sentiment annotation.
                continue

            predefined.add(__to_id(text_opinion))

            text_opinion_linkage = TextOpinionsLinkage([text_opinion])
            text_opinion_linkage.set_tag(service)
            yield text_opinion_linkage


def text_opinion_extraction_pipeline(text_parser, get_doc_func, annotators, terms_per_context, entity_filter):
    assert(callable(get_doc_func))
    assert(isinstance(terms_per_context, int))
    assert(isinstance(entity_filter, EntityFilter))

    return BasePipeline([
        # (doc_id) -> (news)
        MapPipelineItem(map_func=lambda doc_id: get_doc_func(doc_id)),

        # (news) -> (parsed_news)
        MapPipelineItem(map_func=lambda news: NewsParser.parse(news, text_parser)),

        # (parsed_news) -> (text_opinions).
        MapPipelineItem(map_func=lambda parsed_news: iter_text_opinion_linkages(
            annotators=annotators,
            parsed_news=parsed_news,
            terms_per_context=terms_per_context,
            entity_filter=entity_filter)),

        # linkages[] -> linkages.
        FlattenIterPipelineItem()
    ])
