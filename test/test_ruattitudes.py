import unittest

from arekit.common.data.input.writers.tsv import TsvWriter
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.parser import BaseTextParser
from arekit.common.utils import progress_bar_iter
from arekit.contrib.source.ruattitudes.collection import RuAttitudesCollection
from arekit.contrib.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.ruattitudes.news import RuAttitudesNews
from arekit.contrib.source.ruattitudes.news_brat import RuAttitudesNewsConverter
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection

from labels.scaler import PosNegNeuRelationsLabelScaler
from models.nn.serialize import serialize_nn
from pipelines.train import text_opinions_to_opinion_linkages_pipeline


class RuAttitudesDocumentOperations(DocumentOperations):

    def __init__(self, ru_attitudes):
        assert(isinstance(ru_attitudes, dict))
        super(RuAttitudesDocumentOperations, self).__init__()
        self.__ru_attitudes = ru_attitudes

    def get_doc(self, doc_id):
        assert(isinstance(doc_id, int))
        return self.__ru_attitudes[doc_id]


class TestRuAttitudes(unittest.TestCase):
    """ This test is related to #232 issue of the AREkit
        It is expected that we may adopt existed pipeline,
        based on text-opinions for RuAttitudes collection reading.
    """

    @staticmethod
    def __iter_id_with_news(docs_it, keep_doc_ids_only):
        if keep_doc_ids_only:
            for doc_id in docs_it:
                yield doc_id, None
        else:
            for news in docs_it:
                assert (isinstance(news, RuAttitudesNews))
                yield news.ID, news

    @staticmethod
    def read_ruattitudes_to_brat_in_memory(version, keep_doc_ids_only, doc_id_func, label_scaler, limit=None):
        """ Performs reading of ruattitude formatted documents and
            selection according to 'doc_ids_set' parameter.
        """
        assert (isinstance(version, RuAttitudesVersions))
        assert (isinstance(keep_doc_ids_only, bool))
        assert (callable(doc_id_func))

        it = RuAttitudesCollection.iter_news(version=version,
                                             get_news_index_func=doc_id_func,
                                             return_inds_only=keep_doc_ids_only)

        it_formatted_and_logged = progress_bar_iter(
            iterable=TestRuAttitudes.__iter_id_with_news(docs_it=it, keep_doc_ids_only=keep_doc_ids_only),
            desc="Loading RuAttitudes Collection [{}]".format("doc ids only" if keep_doc_ids_only else "fully"),
            unit='docs')

        d = {}
        for doc_id, news in it_formatted_and_logged:
            assert(isinstance(news, RuAttitudesNews))
            d[doc_id] = RuAttitudesNewsConverter.to_brat_news(news, label_scaler=label_scaler)
            if limit is not None and doc_id > limit:
                break

        return d

    def test_serialize_ruattitudes(self):

        ru_attitudes = TestRuAttitudes.read_ruattitudes_to_brat_in_memory(
            version=RuAttitudesVersions.V20Base,
            doc_id_func=lambda doc_id: doc_id,
            keep_doc_ids_only=False,
            label_scaler=PosNegNeuRelationsLabelScaler(),
            limit=200)

        doc_ops = RuAttitudesDocumentOperations(ru_attitudes)

        text_parser = BaseTextParser(pipeline=[RuAttitudesTextEntitiesParser(),
                                               DefaultTextTokenizer(keep_tokens=True)])

        synonyms = StemmerBasedSynonymCollection(iter_group_values_lists=[],
                                                 stemmer=MystemWrapper(),
                                                 is_read_only=False,
                                                 debug=False)

        pipeline = text_opinions_to_opinion_linkages_pipeline(
            terms_per_context=50,
            get_doc_func=lambda doc_id: doc_ops.get_doc(doc_id),
            text_parser=text_parser,
            neut_annotator=None,
            value_to_group_id_func=lambda value:
                SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                    synonyms=synonyms, value=value))

        data_folding = NoFolding(doc_ids_to_fold=ru_attitudes.keys(),
                                 supported_data_types=[DataType.Train])

        serialize_nn(output_dir="_out/serialize-ruattitudes-nn",
                     split_filepath=None,
                     data_type_pipelines={DataType.Train: pipeline},
                     folding_type=None,
                     data_folding=data_folding,
                     writer=TsvWriter(write_header=True))
