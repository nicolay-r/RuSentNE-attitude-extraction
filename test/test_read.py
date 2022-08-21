import unittest
from arekit.common.text_opinions.base import TextOpinion
from arekit.contrib.source.brat.news import BratNews
from arekit.contrib.source.brat.sentence import BratSentence

from SentiNEREL.reader import SentiNERELDocReader


class TestRead(unittest.TestCase):

    def test(self):
        news = SentiNERELDocReader.read_document(filename="2070_text", doc_id=0)
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

        for brat_relation in news.Relations:
            assert(isinstance(brat_relation, TextOpinion))
            print(brat_relation.SourceId, brat_relation.TargetId, str(type(brat_relation.Sentiment)))


if __name__ == '__main__':
    unittest.main()
