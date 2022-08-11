import unittest

from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.opinions.annot.algo.pair_based import PairBasedOpinionAnnotationAlgorithm
from arekit.common.opinions.annot.algo_based import AlgorithmBasedOpinionAnnotator
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms.base import SynonymsCollection
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.bert.samplers.nli_m import NliMultipleSampleProvider
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper
from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions, RuSentRelIOUtils
from arekit.contrib.source.rusentrel.news_reader import RuSentRelNewsReader
from arekit.contrib.source.rusentrel.synonyms import RuSentRelSynonymsCollectionHelper
from arekit.contrib.utils.pipelines.annot.base import attitude_extraction_default_pipeline
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection

from entity.formatter import CustomEntitiesFormatter
from labels.formatter import PosNegNeuRelationsLabelFormatter
from labels.scaler import PosNegNeuRelationsLabelScaler
from models.bert.serialize import serialize_bert
from models.nn.serialize import serialize_nn
from writers.opennre_json import OpenNREJsonWriter


class RuSentrelDocumentOperations(DocumentOperations):
    """ Limitations: Supported only train/test collections format
    """

    def __init__(self, version, synonyms):
        assert(isinstance(version, RuSentRelVersions))
        assert(isinstance(synonyms, SynonymsCollection))
        super(RuSentrelDocumentOperations, self).__init__()
        self.__version = version
        self.__synonyms = synonyms

    def get_doc(self, doc_id):
        assert (isinstance(doc_id, int))
        return RuSentRelNewsReader.read_document(doc_id=doc_id, synonyms=self.__synonyms, version=self.__version)


class TestRuSentRel(unittest.TestCase):
    """ TODO: This might be a test example for AREkit (utils).
    """

    @staticmethod
    def __create_pipeline(rusentrel_version, text_parser):
        """ Processing pipeline for RuSentRel.
        """

        synonyms = StemmerBasedSynonymCollection(
            iter_group_values_lists=RuSentRelSynonymsCollectionHelper.iter_groups(rusentrel_version),
            stemmer=MystemWrapper(),
            is_read_only=False,
            debug=False)

        doc_ops = RuSentrelDocumentOperations(version=rusentrel_version, synonyms=synonyms)

        annotator = AlgorithmBasedOpinionAnnotator(
            annot_algo=PairBasedOpinionAnnotationAlgorithm(
                dist_in_sents=0,
                dist_in_terms_bound=50,
                label_provider=ConstantLabelProvider(NoLabel())),
            create_empty_collection_func=lambda: OpinionCollection(
                opinions=[],
                synonyms=synonyms,
                error_on_duplicates=True,
                error_on_synonym_end_missed=False),
            get_doc_existed_opinions_func=lambda _: None)

        pipeline = attitude_extraction_default_pipeline(
            annotator=annotator,
            entity_index_func=lambda brat_entity: brat_entity.ID,
            terms_per_context=50,
            get_doc_func=lambda doc_id: doc_ops.get_doc(doc_id),
            text_parser=text_parser,
            value_to_group_id_func=lambda value:
                SynonymsCollectionValuesGroupingProviders.provide_existed_value(synonyms=synonyms, value=value))

        return pipeline

    def test_serialize_bert(self):

        version = RuSentRelVersions.V11

        text_parser = BaseTextParser(pipeline=[BratTextEntitiesParser(),
                                               DefaultTextTokenizer()])

        pipeline, ru_attitudes = self.__create_pipeline(rusentrel_version=version,
                                                        text_parser=text_parser)

        data_folding = NoFolding(doc_ids_to_fold=ru_attitudes.keys(),
                                 supported_data_types=[DataType.Train])

        sample_row_provider = NliMultipleSampleProvider(
            label_scaler=PosNegNeuRelationsLabelScaler(),
            text_b_labels_fmt=PosNegNeuRelationsLabelFormatter(),
            text_terms_mapper=BertDefaultStringTextTermsMapper(
                entity_formatter=CustomEntitiesFormatter(subject_fmt="#S", object_fmt="#O")
            ))

        serialize_bert(output_dir="_out/serialize-rusentrel-bert",
                       terms_per_context=50,
                       split_filepath=None,
                       data_type_pipelines={DataType.Train: pipeline},
                       sample_row_provider=sample_row_provider,
                       folding_type=None,
                       data_folding=data_folding,
                       writer=OpenNREJsonWriter())

    def test_serialize_nn(self):

        version = RuSentRelVersions.V11

        stemmer = MystemWrapper()
        frames_collection = RuSentiFramesCollection.read_collection(
            version=RuSentiFramesVersions.V20,
            labels_fmt=RuSentiFramesLabelsFormatter())
        frame_variant_collection = FrameVariantsCollection()
        frame_variant_collection.fill_from_iterable(
            variants_with_id=frames_collection.iter_frame_id_and_variants(),
            overwrite_existed_variant=True,
            raise_error_on_existed_variant=False)

        text_parser = BaseTextParser(pipeline=[BratTextEntitiesParser(),
                                               DefaultTextTokenizer(keep_tokens=True),
                                               LemmasBasedFrameVariantsParser(
                                                   frame_variants=frame_variant_collection,
                                                   stemmer=stemmer)])

        pipeline = self.__create_pipeline(rusentrel_version=version,
                                          text_parser=text_parser)

        data_folding = NoFolding(doc_ids_to_fold=RuSentRelIOUtils.iter_collection_indices(version),
                                 supported_data_types=[DataType.Train])

        serialize_nn(output_dir="_out/serialize-rusentrel-nn",
                     split_filepath=None,
                     data_type_pipelines={DataType.Train: pipeline},
                     folding_type=None,
                     data_folding=data_folding,
                     writer=OpenNREJsonWriter())
