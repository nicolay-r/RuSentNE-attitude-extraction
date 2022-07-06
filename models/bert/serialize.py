from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.experiment.api.ctx_serialization import ExperimentSerializationContext
from arekit.common.experiment.engine import ExperimentEngine
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.nofold import NoFolding
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.bert.handlers.serializer import BertExperimentInputSerializerIterationHandler
from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser

from experiment.doc_ops import CustomDocOperations
from folding.fixed import create_train_test_folding
from models.nn.predict import InferIOUtils
from utils import read_train_test


class BertSerializationContext(ExperimentSerializationContext):

    def __init__(self, label_scaler, terms_per_context, str_entity_formatter,
                 annotator, name_provider, data_folding):
        assert(isinstance(str_entity_formatter, StringEntitiesFormatter))
        assert(isinstance(terms_per_context, int))

        super(BertSerializationContext, self).__init__(annot=annotator,
                                                       name_provider=name_provider,
                                                       label_scaler=label_scaler,
                                                       data_folding=data_folding)

        self.__terms_per_context = terms_per_context
        self.__str_entity_formatter = str_entity_formatter

    @property
    def StringEntityFormatter(self):
        return self.__str_entity_formatter

    @property
    def TermsPerContext(self):
        return self.__terms_per_context


class BertTextsSerializationPipelineItem(BasePipelineItem):

    def __init__(self, fixed_split_filepath, terms_per_context, name_provider, label_formatter,
                 entity_fmt, sample_row_provider, output_dir, limit=None):
        assert(isinstance(limit, int) or limit is None)
        assert(isinstance(fixed_split_filepath, str))
        assert(isinstance(entity_fmt, StringEntitiesFormatter))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(sample_row_provider, BaseSampleRowProvider))
        assert(isinstance(name_provider, ExperimentNameProvider))
        assert(isinstance(output_dir, str))

        self.__exp_ctx = BertSerializationContext(
            label_scaler=SingleLabelScaler(NoLabel()),
            annotator=None,
            terms_per_context=terms_per_context,
            str_entity_formatter=entity_fmt,
            name_provider=name_provider,
            data_folding=NoFolding(doc_ids_to_fold=[], supported_data_types=[]))

        train_filenames, test_filenames = read_train_test(fixed_split_filepath)
        if limit is not None:
            train_filenames = train_filenames[:limit]
            test_filenames = test_filenames[:limit]

        filenames_by_ids, data_folding, etalon_data_folding = create_train_test_folding(
            train_filenames=train_filenames,
            test_filenames=test_filenames)

        self.__exp_ctx.set_data_folding(data_folding)

        self.__exp_io = InferIOUtils(exp_ctx=self.__exp_ctx, output_dir=output_dir)

        text_parser = BaseTextParser(pipeline=[BratTextEntitiesParser()])

        self.__doc_ops = CustomDocOperations(exp_ctx=self.__exp_ctx,
                                             label_formatter=label_formatter,
                                             filename_by_id=filenames_by_ids)

        self.__handler = BertExperimentInputSerializerIterationHandler(
            exp_io=self.__exp_io,
            exp_ctx=self.__exp_ctx,
            doc_ops=self.__doc_ops,
            pipeline=None,
            data_types=None,
            sample_rows_provider=None,
            save_labels_func=None,
            balance_train_samples=True)

    def apply_core(self, input_data, pipeline_ctx):

        engine = ExperimentEngine()
        engine.run([self.__handler])

        return self.__exp_io
