from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.experiment.api.ctx_serialization import ExperimentSerializationContext
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.engine import ExperimentEngine
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.nofold import NoFolding
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.bert.handlers.serializer import BertExperimentInputSerializerIterationHandler
from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser

from annot import create_neutral_annotator
from experiment.doc_ops import CustomDocOperations
from experiment.io import CustomExperimentSerializationIO
from folding.fixed import create_train_test_folding
from models.nn.predict import InferIOUtils
from pipelines.etalon import create_etalon_pipeline
from pipelines.test import create_test_pipeline
from pipelines.train import create_train_pipeline
from utils import read_train_test


class BertSerializationContext(ExperimentSerializationContext):

    def __init__(self, label_scaler, terms_per_context, annotator, name_provider, data_folding):
        assert(isinstance(terms_per_context, int))

        super(BertSerializationContext, self).__init__(annot=annotator,
                                                       name_provider=name_provider,
                                                       label_scaler=label_scaler,
                                                       data_folding=data_folding)

        self.__terms_per_context = terms_per_context

    @property
    def TermsPerContext(self):
        return self.__terms_per_context


class BertTextsSerializationPipelineItem(BasePipelineItem):

    def __init__(self, fixed_split_filepath, terms_per_context, name_provider,
                 label_formatter, sample_row_provider, output_dir, limit=None):
        assert(isinstance(limit, int) or limit is None)
        assert(isinstance(fixed_split_filepath, str))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(sample_row_provider, BaseSampleRowProvider))
        assert(isinstance(name_provider, ExperimentNameProvider))
        assert(isinstance(output_dir, str))

        self.__exp_ctx = BertSerializationContext(
            label_scaler=SingleLabelScaler(NoLabel()),
            annotator=None,
            terms_per_context=terms_per_context,
            name_provider=name_provider,
            data_folding=NoFolding(doc_ids_to_fold=[], supported_data_types=[]))

        train_filenames, test_filenames = read_train_test(fixed_split_filepath)
        if limit is not None:
            train_filenames = train_filenames[:limit]
            test_filenames = test_filenames[:limit]

        filenames_by_ids, data_folding, self.__etalon_data_folding = create_train_test_folding(
            train_filenames=train_filenames,
            test_filenames=test_filenames)

        self.__exp_ctx.set_data_folding(data_folding)

        self.__exp_io = InferIOUtils(exp_ctx=self.__exp_ctx, output_dir=output_dir)

        text_parser = BaseTextParser(pipeline=[BratTextEntitiesParser()])

        doc_ops = CustomDocOperations(exp_ctx=self.__exp_ctx,
                                      label_formatter=label_formatter,
                                      filename_by_id=filenames_by_ids)

        # TODO. Тут идет общая часть с нейросетями, ее можно и нужно вынести.

        train_neut_annot, train_synonyms = create_neutral_annotator(terms_per_context)
        test_neut_annot, test_synonyms = create_neutral_annotator(terms_per_context)
        etalon_neut_annot, etalon_synonyms = create_neutral_annotator(terms_per_context)

        self.__train_test_handler = BertExperimentInputSerializerIterationHandler(
            doc_ops=doc_ops,
            balance_train_samples=True,
            exp_io=self.__exp_io,
            exp_ctx=self.__exp_ctx,
            data_type_pipelines={
                DataType.Train: create_train_pipeline(text_parser=text_parser,
                                                      doc_ops=doc_ops,
                                                      neut_annotator=train_neut_annot,
                                                      synonyms=train_synonyms,
                                                      terms_per_context=terms_per_context),
                DataType.Test: create_test_pipeline(text_parser=text_parser,
                                                    doc_ops=doc_ops,
                                                    neut_annotator=test_neut_annot,
                                                    synonyms=test_synonyms,
                                                    terms_per_context=terms_per_context)
            },
            save_labels_func=lambda data_type: data_type == DataType.Train,
            sample_rows_provider=sample_row_provider)

        self.__etalon_handler = BertExperimentInputSerializerIterationHandler(
            doc_ops=doc_ops,
            balance_train_samples=False,
            exp_io=CustomExperimentSerializationIO(output_dir=output_dir, exp_ctx=self.__exp_ctx),
            data_type_pipelines={
                DataType.Etalon: create_etalon_pipeline(text_parser=text_parser,
                                                        doc_ops=doc_ops,
                                                        synonyms=etalon_synonyms,
                                                        terms_per_context=terms_per_context)
            },
            sample_rows_provider=sample_row_provider,
            save_labels_func=lambda data_type: data_type == DataType.Etalon,
            exp_ctx=self.__exp_ctx)

    def apply_core(self, input_data, pipeline_ctx):

        engine = ExperimentEngine()
        engine.run(states_iter=[0], handlers=[self.__train_test_handler])
        self.__exp_ctx.set_data_folding(self.__etalon_data_folding)
        engine.run(states_iter=[0], handlers=[self.__etalon_handler])

        return self.__exp_io
