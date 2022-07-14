from arekit.common.data import const
from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.entities.base import Entity
from arekit.common.experiment.api.ctx_serialization import ExperimentSerializationContext
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.engine import ExperimentEngine
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.bert.handlers.serializer import BertExperimentInputSerializerIterationHandler
from arekit.contrib.bert.samplers.nli_m import NliMultipleSampleProvider
from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.processing.text.pipeline_tokenizer import DefaultTextTokenizer

from folding.factory import FoldingFactory
from labels.formatter import SentimentLabelFormatter
from models.nn.predict import InferIOUtils
from pipelines.collection import prepare_data_pipelines


class BertSerializationContext(ExperimentSerializationContext):

    def __init__(self, label_scaler, terms_per_context, name_provider):
        assert(isinstance(terms_per_context, int))

        super(BertSerializationContext, self).__init__(name_provider=name_provider,
                                                       label_scaler=label_scaler)

        self.__terms_per_context = terms_per_context

    @property
    def TermsPerContext(self):
        return self.__terms_per_context


class CroppedBertSampleRowProvider(NliMultipleSampleProvider):
    """ Нужно немного изменить базовый провайдер так, чтобы
        возвращался небольшой контекст, который влкючает в себя
        объект и субъект, чтобы когда на вход в BERT будут подаваться семплы,
        не возникло проблемы отсечения ввиду огромного предложения.
    """

    def __init__(self, crop_window_size, label_scaler, text_b_labels_fmt, text_terms_mapper):
        super(CroppedBertSampleRowProvider, self).__init__(label_scaler=label_scaler,
                                                           text_b_labels_fmt=text_b_labels_fmt,
                                                           text_terms_mapper=text_terms_mapper)
        self.__crop_window_size = crop_window_size

    @staticmethod
    def __calc_window_bounds(window_size, s_ind, t_ind, input_length):
        """ returns: [_from, _to)
        """
        assert(isinstance(s_ind, int))
        assert(isinstance(t_ind, int))
        assert(isinstance(input_length, int))
        assert(input_length >= s_ind and input_length >= t_ind)

        def __in():
            return _from <= s_ind < _to and _from <= t_ind < _to

        _from = 0
        _to = window_size
        while not __in():
            _from += 1
            _to += 1

        return _from, _to

    def _fill_row_core(self, row, text_opinion_linkage, index_in_linked, etalon_label,
                       parsed_news, sentence_ind, s_ind, t_ind):

        def __assign_value(column, value):
            row[column] = value

        super(CroppedBertSampleRowProvider, self)._fill_row_core(row=row,
                                                                 text_opinion_linkage=text_opinion_linkage,
                                                                 index_in_linked=index_in_linked,
                                                                 etalon_label=etalon_label,
                                                                 parsed_news=parsed_news,
                                                                 sentence_ind=sentence_ind,
                                                                 s_ind=s_ind,
                                                                 t_ind=t_ind)

        # вырезаем часть текста.

        _from, _to = self.__calc_window_bounds(window_size=self.__crop_window_size,
                                               s_ind=s_ind, t_ind=t_ind,
                                               input_length=len(row["text_a"]))

        expected_label = text_opinion_linkage.get_linked_label()

        sentence_terms = list(self._provide_sentence_terms(parsed_news=parsed_news, sentence_ind=sentence_ind))

        cropped_sentence_terms = sentence_terms[_from:_to]
        s_ind = s_ind - _from
        t_ind = t_ind - _from

        self.TextProvider.add_text_in_row(
            set_text_func=lambda column, value: __assign_value(column, value),
            sentence_terms=cropped_sentence_terms,
            s_ind=s_ind,
            t_ind=t_ind,
            expected_label=expected_label)

        # обновляем содержимое.
        entities_in_cropped = list(filter(lambda term: isinstance(term, Entity), cropped_sentence_terms))

        cropped_entity_ids = [str(i) for i, term in enumerate(cropped_sentence_terms) if isinstance(term, Entity)]

        row[const.ENTITY_VALUES] = ",".join([e.Value.replace(',', '') for e in entities_in_cropped])
        row[const.ENTITY_TYPES] = ",".join([e.Type.replace(',', '') for e in entities_in_cropped])
        row[const.ENTITIES] = ",".join(cropped_entity_ids)

        row[const.S_IND] = s_ind
        row[const.T_IND] = t_ind


class BertTextsSerializationPipelineItem(BasePipelineItem):

    def __init__(self, split_filepath, terms_per_context, name_provider,
                 sample_row_provider, output_dir, folding_type="fixed", limit=None):
        assert(isinstance(limit, int) or limit is None)
        assert(isinstance(split_filepath, str))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(sample_row_provider, BaseSampleRowProvider))
        assert(isinstance(name_provider, ExperimentNameProvider))
        assert(isinstance(output_dir, str))

        self.__exp_ctx = BertSerializationContext(label_scaler=SingleLabelScaler(NoLabel()),
                                                  terms_per_context=terms_per_context,
                                                  name_provider=name_provider)
        data_folding = None
        doc_ops = None

        if folding_type == "fixed":
            data_folding, doc_ops = FoldingFactory.create_fixed_folding(
                fixed_split_filepath=split_filepath,
                label_formatter=SentimentLabelFormatter(),
                limit=limit)

        self.__exp_io = InferIOUtils(exp_ctx=self.__exp_ctx, output_dir=output_dir)

        text_parser = BaseTextParser(pipeline=[BratTextEntitiesParser(),
                                               DefaultTextTokenizer()])

        self.__handler = BertExperimentInputSerializerIterationHandler(
            doc_ops=doc_ops,
            balance_func=lambda data_type: data_type == DataType.Train,
            exp_io=self.__exp_io,
            data_folding=data_folding,
            data_type_pipelines=prepare_data_pipelines(
                text_parser=text_parser, doc_ops=doc_ops, terms_per_context=terms_per_context),
            save_labels_func=lambda data_type: data_type != DataType.Test,
            sample_rows_provider=sample_row_provider)

    def apply_core(self, input_data, pipeline_ctx):
        engine = ExperimentEngine()
        engine.run(states_iter=[0], handlers=[self.__handler])
        return self.__exp_io
