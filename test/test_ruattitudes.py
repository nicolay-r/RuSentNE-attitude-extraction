import unittest

from arekit.common.data.input.writers.tsv import TsvWriter
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.parser import BaseTextParser
from arekit.common.utils import progress_bar_iter
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper
from arekit.contrib.source.ruattitudes.collection import RuAttitudesCollection
from arekit.contrib.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.ruattitudes.news import RuAttitudesNews
from arekit.contrib.source.ruattitudes.news_brat import RuAttitudesNewsConverter
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection

from entity.formatter import CustomEntitiesFormatter
from labels.formatter import PosNegNeuRelationsLabelFormatter
from labels.scaler import PosNegNeuRelationsLabelScaler
from models.bert.serialize import serialize_bert, CroppedBertSampleRowProvider
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
        TODO. This test could be moved into AREkit. (#232)
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

    @staticmethod
    def __create_pipeline(text_parser):
        """ Processing pipeline for RuAttitudes.
        """

        synonyms = StemmerBasedSynonymCollection(iter_group_values_lists=[],
                                                 stemmer=MystemWrapper(),
                                                 is_read_only=False,
                                                 debug=False)

        ru_attitudes = TestRuAttitudes.read_ruattitudes_to_brat_in_memory(
            version=RuAttitudesVersions.V20Base,
            doc_id_func=lambda doc_id: doc_id,
            keep_doc_ids_only=False,
            label_scaler=PosNegNeuRelationsLabelScaler(),
            limit=200)

        doc_ops = RuAttitudesDocumentOperations(ru_attitudes)

        pipeline = text_opinions_to_opinion_linkages_pipeline(
            terms_per_context=50,
            get_doc_func=lambda doc_id: doc_ops.get_doc(doc_id),
            text_parser=text_parser,
            neut_annotator=None,
            value_to_group_id_func=lambda value:
            SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                synonyms=synonyms, value=value))

        return pipeline, ru_attitudes

    def test_serialize_bert(self):

        text_parser = BaseTextParser(pipeline=[RuAttitudesTextEntitiesParser(),
                                               DefaultTextTokenizer()])

        pipeline, ru_attitudes = self.__create_pipeline(text_parser=text_parser)

        data_folding = NoFolding(doc_ids_to_fold=ru_attitudes.keys(),
                                 supported_data_types=[DataType.Train])

        sample_row_provider = CroppedBertSampleRowProvider(
            crop_window_size=50,
            label_scaler=PosNegNeuRelationsLabelScaler(),
            text_b_labels_fmt=PosNegNeuRelationsLabelFormatter(),
            text_terms_mapper=BertDefaultStringTextTermsMapper(
                entity_formatter=CustomEntitiesFormatter(subject_fmt="#S", object_fmt="#O")
        ))

        serialize_bert(output_dir="_out/serialize-ruattitudes-bert",
                       terms_per_context=50,
                       split_filepath=None,
                       data_type_pipelines={DataType.Train: pipeline},
                       sample_row_provider=sample_row_provider,
                       folding_type=None,
                       data_folding=data_folding,
                       writer=TsvWriter(write_header=True))

    def test_serialize_nn(self):

        stemmer = MystemWrapper()
        frames_collection = RuSentiFramesCollection.read_collection(
            version=RuSentiFramesVersions.V20,
            labels_fmt=RuSentiFramesLabelsFormatter())
        frame_variant_collection = FrameVariantsCollection()
        frame_variant_collection.fill_from_iterable(
            variants_with_id=frames_collection.iter_frame_id_and_variants(),
            overwrite_existed_variant=True,
            raise_error_on_existed_variant=False)

        text_parser = BaseTextParser(pipeline=[RuAttitudesTextEntitiesParser(),
                                               DefaultTextTokenizer(keep_tokens=True),
                                               LemmasBasedFrameVariantsParser(
                                                   frame_variants=frame_variant_collection,
                                                   stemmer=stemmer)])

        pipeline, ru_attitudes = self.__create_pipeline(text_parser=text_parser)

        data_folding = NoFolding(doc_ids_to_fold=ru_attitudes.keys(),
                                 supported_data_types=[DataType.Train])

        serialize_nn(output_dir="_out/serialize-ruattitudes-nn",
                     split_filepath=None,
                     data_type_pipelines={DataType.Train: pipeline},
                     folding_type=None,
                     data_folding=data_folding,
                     writer=TsvWriter(write_header=True))
