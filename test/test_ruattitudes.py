import unittest

from arekit.common.data.input.writers.tsv import TsvWriter
from arekit.common.entities.base import Entity
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.entities.types import OpinionEntityType
from arekit.common.experiment.api.ops_doc import DocumentOperations
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.parser import BaseTextParser
from arekit.common.utils import progress_bar_iter
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper
from arekit.contrib.source.ruattitudes.collection import RuAttitudesCollection
from arekit.contrib.source.ruattitudes.entity.parser import RuAttitudesTextEntitiesParser
from arekit.contrib.source.ruattitudes.io_utils import RuAttitudesVersions
from arekit.contrib.source.ruattitudes.news import RuAttitudesNews
from arekit.contrib.source.ruattitudes.news_brat import RuAttitudesNewsConverter
from arekit.contrib.source.ruattitudes.synonyms import RuAttitudesSynonymsCollectionHelper
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection

from entity.filter import EntityFilter
from labels.formatter import PosNegNeuRelationsLabelFormatter
from labels.scaler import PosNegNeuRelationsLabelScaler
from models.bert.serialize import serialize_bert, CroppedBertSampleRowProvider
from models.nn.serialize import serialize_nn
from pipelines.train import text_opinions_to_opinion_linkages_pipeline
from writers.opennre_json import OpenNREJsonWriter


class DictionaryBasedDocumentOperations(DocumentOperations):

    def __init__(self, ru_attitudes):
        assert(isinstance(ru_attitudes, dict))
        super(DictionaryBasedDocumentOperations, self).__init__()
        self.__ru_attitudes = ru_attitudes

    def get_doc(self, doc_id):
        assert(isinstance(doc_id, int))
        return self.__ru_attitudes[doc_id]


class RuAttitudesEntityFilter(EntityFilter):

    def is_ignored(self, entity, e_type):

        supported = ["GPE", "PERSON", "LOCAL", "GEO", "ORG"]

        if e_type == OpinionEntityType.Subject:
            return entity.Type not in supported
        if e_type == OpinionEntityType.Object:
            return entity.Type not in supported
        else:
            return True


class RuAttitudesEntitiesFormatter(StringEntitiesFormatter):
    """ Форматирование сущностей. Было принято решение использовать тип сущности в качетстве значений.
        Поскольку тексты русскоязычные, то и типы были руссифицированы из соображений более удачных embeddings.
    """

    def __init__(self, subject_fmt='[субъект]', object_fmt="[объект]"):
        self.__subject_fmt = subject_fmt
        self.__object_fmt = object_fmt

    def to_string(self, original_value, entity_type):
        assert(isinstance(original_value, Entity))
        assert(isinstance(entity_type, OpinionEntityType))

        if entity_type == OpinionEntityType.Other:
            return original_value.Type
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
        """ Performs reading of RuAttitude formatted documents and
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
        docs_read = 0
        for doc_id, news in it_formatted_and_logged:
            assert(isinstance(news, RuAttitudesNews))
            d[doc_id] = RuAttitudesNewsConverter.to_brat_news(news, label_scaler=label_scaler)
            docs_read += 1
            if limit is not None and docs_read >= limit:
                break

        return d

    @staticmethod
    def __create_pipeline(text_parser,
                          label_scaler,
                          version=RuAttitudesVersions.V20Large,
                          terms_per_context=50,
                          entity_filter=RuAttitudesEntityFilter(),
                          nolabel_annotator=None,
                          limit=None):
        """ Processing pipeline for RuAttitudes.
            This pipeline is based on the in-memory RuAttitudes storage.

            version: enum
                Version of the RuAttitudes collection.
                NOTE: we consider to support a variations of the 2.0 versions.
            label_scaler:
                Scaler that allows to perform conversion from integer labels (RuAttitudes) to
                the actual `Label` instances, required in further for text_opinions instances.
            terms_per_context: int
                Amount of terms that we consider in between the Object and Subject.
            entity_filter:
                Entity filter
            nolabel_annotator:
                Annontator that could be adopted in order to perform automatic annotation
                of a non-labeled attitudes.
            limit: int or None
                Limit of documents to consider.
        """
        assert(isinstance(label_scaler, BaseLabelScaler))
        assert(isinstance(version, RuAttitudesVersions))

        synonyms = StemmerBasedSynonymCollection(
            iter_group_values_lists=RuAttitudesSynonymsCollectionHelper.iter_groups(version),
            stemmer=MystemWrapper(),
            is_read_only=False,
            debug=False)

        ru_attitudes = TestRuAttitudes.read_ruattitudes_to_brat_in_memory(
            version=version,
            doc_id_func=lambda doc_id: doc_id,
            keep_doc_ids_only=False,
            label_scaler=label_scaler,
            limit=limit)

        doc_ops = DictionaryBasedDocumentOperations(ru_attitudes)

        pipeline = text_opinions_to_opinion_linkages_pipeline(
            terms_per_context=terms_per_context,
            get_doc_func=lambda doc_id: doc_ops.get_doc(doc_id),
            text_parser=text_parser,
            neut_annotator=nolabel_annotator,
            entity_filter=entity_filter,
            value_to_group_id_func=lambda value:
            SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                synonyms=synonyms, value=value))

        return pipeline, ru_attitudes

    def __test_serialize_bert(self, writer):

        text_parser = BaseTextParser(pipeline=[RuAttitudesTextEntitiesParser(),
                                               DefaultTextTokenizer()])

        pipeline, ru_attitudes = self.__create_pipeline(text_parser=text_parser,
                                                        label_scaler=PosNegNeuRelationsLabelScaler())

        data_folding = NoFolding(doc_ids_to_fold=ru_attitudes.keys(),
                                 supported_data_types=[DataType.Train])

        sample_row_provider = CroppedBertSampleRowProvider(
            crop_window_size=50,
            label_scaler=PosNegNeuRelationsLabelScaler(),
            text_b_labels_fmt=PosNegNeuRelationsLabelFormatter(),
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

        pipeline, ru_attitudes = self.__create_pipeline(text_parser=text_parser,
                                                        label_scaler=PosNegNeuRelationsLabelScaler())

        data_folding = NoFolding(doc_ids_to_fold=ru_attitudes.keys(),
                                 supported_data_types=[DataType.Train])

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
