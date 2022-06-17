from arekit.common.news.base import News
from arekit.contrib.source.brat.entities.entity import BratEntity
from arekit.contrib.source.brat.sentence import BratSentence


class CustomNews(News):

    def __init__(self, doc_id, sentences, text_opinions):
        assert(isinstance(text_opinions, list))
        super(CustomNews, self).__init__(doc_id=doc_id,
                                         sentences=sentences)

        self.__text_opinions = text_opinions

        self.__entity_by_id = {}

        for s in sentences:
            assert(isinstance(s, BratSentence))
            for e, _ in s.iter_entity_with_local_bounds():
                assert(isinstance(e, BratEntity))
                self.__entity_by_id[e.ID] = e

    @property
    def TextOpinions(self):
        for o in self.__text_opinions:
            yield o

    def contains_entity(self, entity_id):
        return entity_id in self.__entity_by_id

    def get_entity_by_id(self, entity_id):
        return self.__entity_by_id[entity_id]
