from arekit.common.data.input.providers.rows.samples import BaseSampleRowProvider
from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.base import BasePipeline
from arekit.common.text.parser import BaseTextParser

from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.pipelines.items.sampling.bert import BertExperimentInputSerializerPipelineItem
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer

from SentiNEREL.doc_ops import CollectionDocOperation
from SentiNEREL.folding.factory import FoldingFactory
from SentiNEREL.labels.formatter import SentimentLabelFormatter
from SentiNEREL.pipelines.data import prepare_data_pipelines


def serialize_bert(split_filepath, terms_per_context, writer, sample_row_provider, output_dir,
                   data_type_pipelines=None, data_folding=None, folding_type="fixed", limit=None):
    assert(isinstance(limit, int) or limit is None)
    assert(isinstance(split_filepath, str) or split_filepath is None)
    assert(isinstance(terms_per_context, int))
    assert(isinstance(sample_row_provider, BaseSampleRowProvider))
    assert(isinstance(output_dir, str))

    pipeline = BasePipeline([
        BertExperimentInputSerializerPipelineItem(
            balance_func=lambda data_type: data_type == DataType.Train,
            samples_io=SamplesIO(target_dir=output_dir, writer=writer),
            save_labels_func=lambda data_type: data_type != DataType.Test,
            sample_rows_provider=sample_row_provider)
    ])

    doc_ops = None

    if data_folding is None:
        # Selecting from presets.
        if folding_type == "fixed":
            filenames_by_ids, data_folding = FoldingFactory.create_fixed_folding(
                fixed_split_filepath=split_filepath, limit=limit)
            doc_ops = CollectionDocOperation(filenames_by_ids)

    if data_type_pipelines is None:
        # considering a default pipeline.
        text_parser = BaseTextParser(pipeline=[BratTextEntitiesParser(),
                                               DefaultTextTokenizer()])
        data_type_pipelines = prepare_data_pipelines(text_parser=text_parser,
                                                     doc_ops=doc_ops,
                                                     label_formatter=SentimentLabelFormatter(),
                                                     terms_per_context=terms_per_context)

    pipeline.run(input_data=None,
                 params_dict={
                     "data_folding": data_folding,
                     "data_type_pipelines": data_type_pipelines
                 })
