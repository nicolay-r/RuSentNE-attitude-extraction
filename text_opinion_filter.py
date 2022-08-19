from arekit.common.data.input.sample import InputSampleBase
from arekit.common.entities.types import OpinionEntityType
from arekit.common.news.parsed.base import ParsedNews
from arekit.common.news.parsed.term_position import TermPositionTypes
from arekit.common.text_opinions.base import TextOpinion


class TextOpinionFilter(object):

    def filter(self, text_opinion, parsed_news, entity_service_provider):
        raise NotImplementedError()


class FrameworkLimitationsTextOpinionFilter(TextOpinionFilter):

    def filter(self, text_opinion, parsed_news, entity_service_provider):
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(parsed_news, ParsedNews))

        s_ind = entity_service_provider.get_entity_position(
            text_opinion.SourceId, position_type=TermPositionTypes.SentenceIndex)
        t_ind = entity_service_provider.get_entity_position(
            text_opinion.TargetId, position_type=TermPositionTypes.SentenceIndex)

        if s_ind != t_ind:
            # AREkit does not provide a support for multi-sentence opinions at present.
            return False

        return True


class EntityBasedTextOpinionFilter(TextOpinionFilter):

    def __init__(self, entity_filter):
        super(EntityBasedTextOpinionFilter, self).__init__()
        self.__entity_filter = entity_filter

    def filter(self, text_opinion, parsed_news, entity_service_provider):
        assert(isinstance(text_opinion, TextOpinion))
        assert(isinstance(parsed_news, ParsedNews))

        e_source = entity_service_provider._doc_entities[text_opinion.SourceId]
        if self.__entity_filter is not None and self.__entity_filter.is_ignored(e_source, OpinionEntityType.Subject):
            return False

        e_target = entity_service_provider._doc_entities[text_opinion.TargetId]
        if self.__entity_filter is not None and self.__entity_filter.is_ignored(e_target, OpinionEntityType.Object):
            return False

        return True


class DistanceLimitedTextOpinionFilter(TextOpinionFilter):

    def __init__(self, terms_per_context):
        super(DistanceLimitedTextOpinionFilter, self).__init__()
        self.__terms_per_context = terms_per_context

    def filter(self, text_opinion, parsed_news, entity_service_provider):

        return InputSampleBase.check_ability_to_create_sample(
            entity_service=entity_service_provider,
            text_opinion=text_opinion,
            window_size=self.__terms_per_context)
