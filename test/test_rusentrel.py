import unittest

from arekit.common.data.input.writers.tsv import TsvWriter
from arekit.common.entities.base import Entity
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import OpinionEntityType
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


class RuSentRelEntitiesFormatter(StringEntitiesFormatter):
    """ Форматирование сущностей. Было принято решение использовать тип сущности в качетстве значений.
        Поскольку тексты русскоязычные, то и типы были руссифицированы из соображений более удачных embeddings.
    """

    type_formatter = {
        "GEOPOLIT": "гео-сущность",
        "ORG": "организация",
        "PER": "личность",
        "LOC": "локация",
        "ОRG": "организация"
    }

    def __init__(self, subject_fmt='[субъект]', object_fmt="[объект]"):
        self.__subject_fmt = subject_fmt
        self.__object_fmt = object_fmt

    def to_string(self, original_value, entity_type):
        assert(isinstance(original_value, Entity))
        assert(isinstance(entity_type, OpinionEntityType))

        if entity_type == OpinionEntityType.Other:
            return RuSentRelEntitiesFormatter.type_formatter[original_value.Type]
        elif entity_type == OpinionEntityType.Object or entity_type == OpinionEntityType.SynonymObject:
            return self.__object_fmt
        elif entity_type == OpinionEntityType.Subject or entity_type == OpinionEntityType.SynonymSubject:
            return self.__subject_fmt

        return None


class TestRuSentRel(unittest.TestCase):
    """ TODO: This might be a test example for AREkit (utils).
    """

    @staticmethod
    def __create_pipeline(rusentrel_version, text_parser, terms_per_context=50, dist_in_sentences=0):
        """ Processing pipeline for RuSentRel.

            Original collection paper: arxiv.org/abs/1808.08932

            version: enum
                Version of the RuSentRel collection.
            terms_per_context: int
                Amount of terms that we consider in between the Object and Subject.
            dist_in_sentences: int
                considering amount of sentences that could be in between Object and Subject.
        """

        synonyms = StemmerBasedSynonymCollection(
            iter_group_values_lists=RuSentRelSynonymsCollectionHelper.iter_groups(rusentrel_version),
            stemmer=MystemWrapper(),
            is_read_only=False,
            debug=False)

        doc_ops = RuSentrelDocumentOperations(version=rusentrel_version, synonyms=synonyms)

        annotator = AlgorithmBasedOpinionAnnotator(
            annot_algo=PairBasedOpinionAnnotationAlgorithm(
                dist_in_sents=dist_in_sentences,
                dist_in_terms_bound=terms_per_context,
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
            terms_per_context=terms_per_context,
            get_doc_func=lambda doc_id: doc_ops.get_doc(doc_id),
            text_parser=text_parser,
            value_to_group_id_func=lambda value:
                SynonymsCollectionValuesGroupingProviders.provide_existed_value(synonyms=synonyms, value=value))

        return pipeline

    def __test_serialize_bert(self, writer):

        version = RuSentRelVersions.V11

        text_parser = BaseTextParser(pipeline=[BratTextEntitiesParser(),
                                               DefaultTextTokenizer()])

        pipeline = self.__create_pipeline(rusentrel_version=version,
                                          text_parser=text_parser)

        data_folding = NoFolding(doc_ids_to_fold=RuSentRelIOUtils.iter_collection_indices(version),
                                 supported_data_types=[DataType.Train])

        sample_row_provider = NliMultipleSampleProvider(
            label_scaler=PosNegNeuRelationsLabelScaler(),
            text_b_labels_fmt=PosNegNeuRelationsLabelFormatter(),
            text_terms_mapper=BertDefaultStringTextTermsMapper(
                entity_formatter=RuSentRelEntitiesFormatter(subject_fmt="#S", object_fmt="#O")
            ))

        serialize_bert(output_dir="_out/serialize-rusentrel-bert",
                       terms_per_context=50,
                       split_filepath=None,
                       data_type_pipelines={DataType.Train: pipeline},
                       sample_row_provider=sample_row_provider,
                       folding_type=None,
                       data_folding=data_folding,
                       writer=writer)

    def __test_serialize_nn(self, writer):

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
                     writer=writer)

    def test_serialize_bert_csv(self):
        self.__test_serialize_bert(writer=TsvWriter(write_header=True))

    def test_serialize_bert_opennre(self):
        self.__test_serialize_bert(writer=OpenNREJsonWriter("bert"))

    def test_serialize_nn_csv(self):
        self.__test_serialize_nn(writer=TsvWriter(write_header=True))

    def test_serialize_nn_opennre(self):
        self.__test_serialize_nn(writer=OpenNREJsonWriter())
