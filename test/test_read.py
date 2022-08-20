import unittest
from arekit.common.text_opinions.base import TextOpinion
from arekit.contrib.source.brat.news import BratNews
from arekit.contrib.source.brat.sentence import BratSentence

from SentiNEREL.reader import SentiNERELDocReader
from labels.formatter import SentimentLabelFormatter


class TestRead(unittest.TestCase):

    def test(self):
        news = SentiNERELDocReader.read_document(filename="2070_text", doc_id=0,
                                                 label_formatter=SentimentLabelFormatter(),
                                                 keep_any_opinion=False)
        assert(isinstance(news, BratNews))
        print("Sentences Count:", news.SentencesCount)
        for sentence in news.iter_sentences():
            assert(isinstance(sentence, BratSentence))
            print(sentence.Text.strip())
            for entity, bound in sentence.iter_entity_with_local_bounds():
                print("{}: ['{}',{}, {}]".format(
                    entity.ID, entity.Value, entity.Type,
                    "-".join([str(bound.Position), str(bound.Position+bound.Length)])))

        print()

        for text_opinion in news.TextOpinions:
            assert(isinstance(text_opinion, TextOpinion))
            print(text_opinion.SourceId, text_opinion.TargetId, str(type(text_opinion.Sentiment)))


if __name__ == '__main__':
    unittest.main()
