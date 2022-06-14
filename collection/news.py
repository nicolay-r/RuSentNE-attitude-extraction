from arekit.common.news.base import News


class CustomNews(News):

    def __init__(self, doc_id, sentences, text_opinions):
        assert(isinstance(text_opinions, list))
        super(CustomNews, self).__init__(doc_id=doc_id,
                                         sentences=sentences)

        self.__text_opinions = text_opinions

    @property
    def TextOpinions(self):
        for o in self.__text_opinions:
            yield o
