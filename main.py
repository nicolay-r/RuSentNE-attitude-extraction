from arekit.common.news.base import News
from arekit.common.text_opinions.base import TextOpinion

from collection.opinions.label_fmt import CustomLabelFormatter
from collection.reader import CollectionNewsReader


if __name__ == '__main__':
     doc_id = 15088

     label_formatter = CustomLabelFormatter()

     doc = CollectionNewsReader.read_document(doc_id=doc_id, label_formatter=label_formatter)
     assert(isinstance(doc, News))

     for o in doc.TextOpinions:
         assert(isinstance(o, TextOpinion))
         print(o.Owner, o.SourceId, o.TargetId, o.Sentiment)

     print(doc.SentencesCount)
