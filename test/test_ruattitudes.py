import unittest
from os.path import join

from arekit.common.data.input.writers.tsv import TsvWriter
from arekit.common.entities.base import Entity
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import OpinionEntityType
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper
from arekit.contrib.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter, \
    RuSentiFramesEffectLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.utils.bert.text_b_rus import BertTextBTemplates
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.pipelines.sources.ruattitudes.extract_text_opinions import \
    create_text_opinion_extraction_pipeline
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper

from SentiNEREL.labels.scaler import PosNegNeuRelationsLabelScaler
from SentiNEREL.labels.types import PositiveTo, NegativeTo
from __run_evaluation import show_stat_for_samples
from models.bert.serialize import serialize_bert, CroppedBertSampleRowProvider
from models.nn.serialize import serialize_nn
from writers.opennre_json import OpenNREJsonWriter


class RuAttitudesEntitiesFormatter(StringEntitiesFormatter):
    """ Форматирование сущностей. Было принято решение использовать тип сущности в качетстве значений.
        Поскольку тексты русскоязычные, то и типы были руссифицированы из соображений более удачных embeddings.
    """

    type_formatter = {
        "GPE": "гео-сущность",
        "PERSON": "личность",
        "LOCAL": "локация",
        "ОRG": "организация"
    }

    def __init__(self, subject_fmt='[субъект]', object_fmt="[объект]"):
        self.__subject_fmt = subject_fmt
        self.__object_fmt = object_fmt

    def to_string(self, original_value, entity_type):
        assert(isinstance(original_value, Entity))
        assert(isinstance(entity_type, OpinionEntityType))

        if entity_type == OpinionEntityType.Other:
            return self.type_formatter[original_value.Type] \
                if original_value.Type in self.type_formatter else original_value.Value
        elif entity_type == OpinionEntityType.Object or entity_type == OpinionEntityType.SynonymObject:
            return self.__object_fmt
        elif entity_type == OpinionEntityType.Subject or entity_type == OpinionEntityType.SynonymSubject:
            return self.__subject_fmt

        return None


class TestRuAttitudes(unittest.TestCase):
    """ This test is related to #232 issue of the AREkit
        It is expected that we may adopt existed pipeline,
        based on text-opinions for RuAttitudes collection reading.
        TODO. This test could be moved into AREkit. (#232)
    """

    def __test_serialize_bert(self, writer):

        text_parser = BaseTextParser(pipeline=[RuAttitudesTextEntitiesParser(),
                                               DefaultTextTokenizer()])

        pipeline, ru_attitudes = create_text_opinion_extraction_pipeline(
            text_parser=text_parser, label_scaler=PosNegNeuRelationsLabelScaler())

        data_folding = NoFolding(doc_ids=ru_attitudes.keys(),
                                 supported_data_type=[DataType.Train])

        sample_row_provider = CroppedBertSampleRowProvider(
            crop_window_size=50,
            label_scaler=PosNegNeuRelationsLabelScaler(),
            text_b_template=BertTextBTemplates.NLI.value,
            text_terms_mapper=BertDefaultStringTextTermsMapper(
                entity_formatter=RuAttitudesEntitiesFormatter(subject_fmt="#S", object_fmt="#O")
        ))

        serialize_bert(output_dir="_out/serialize-ruattitudes-bert",
                       terms_per_context=50,
                       split_filepath=None,
                       data_type_pipelines={DataType.Train: pipeline},
                       sample_row_provider=sample_row_provider,
                       folding_type=None,
                       data_folding=data_folding,
                       writer=writer)

    def __test_serialize_nn(self, writer):

        stemmer = MystemWrapper()
        frames_collection = RuSentiFramesCollection.read_collection(
            version=RuSentiFramesVersions.V20,
            labels_fmt=RuSentiFramesLabelsFormatter(pos_label_type=PositiveTo, neg_label_type=NegativeTo),
            effect_labels_fmt=RuSentiFramesEffectLabelsFormatter(pos_label_type=PositiveTo, neg_label_type=NegativeTo))
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

        pipeline, ru_attitudes = create_text_opinion_extraction_pipeline(
            text_parser=text_parser, label_scaler=PosNegNeuRelationsLabelScaler())

        data_folding = NoFolding(doc_ids=ru_attitudes.keys(),
                                 supported_data_type=DataType.Train)

        serialize_nn(output_dir="_out/serialize-ruattitudes-nn",
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

    def test_show_stat(self):
        show_stat_for_samples(samples_filepath=join("_out/serialize-ruattitudes-bert", "sample-train-0.tsv.gz"),
                              no_label_uint=0)
