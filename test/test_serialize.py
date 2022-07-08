import unittest
from os.path import dirname, realpath, join

from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.bert.samplers.nli_m import NliMultipleSampleProvider
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper

from entity.formatter import CustomEntitiesFormatter
from labels.formatter import PosNegNeuRelationsLabelFormatter, SentimentLabelFormatter
from labels.scaler import PosNegNeuRelationsLabelScaler
from models.bert.serialize import BertTextsSerializationPipelineItem
from models.nn.serialize import serialize_nn


class TestSerialize(unittest.TestCase):

    def test_nn(self):

        current_dir = dirname(realpath(__file__))
        output_dir = join(current_dir, "_out")

        serialize_nn(limit=1, output_dir=output_dir, fixed_split_filepath="../data/split_fixed.txt")

    def test_bert(self):

        current_dir = dirname(realpath(__file__))
        output_dir = join(current_dir, "_out")

        ppl = BasePipeline([
            BertTextsSerializationPipelineItem(
                limit=1,
                terms_per_context=50,
                output_dir=output_dir,
                fixed_split_filepath="../data/split_fixed.txt",
                name_provider=ExperimentNameProvider(name="serialize", suffix="bert"),
                label_formatter=SentimentLabelFormatter(),
                sample_row_provider=NliMultipleSampleProvider(
                    label_scaler=PosNegNeuRelationsLabelScaler(),
                    text_b_labels_fmt=PosNegNeuRelationsLabelFormatter(),
                    text_terms_mapper=BertDefaultStringTextTermsMapper(
                        entity_formatter=CustomEntitiesFormatter(subject_fmt="#S", object_fmt="#O")
                    )))
        ])

        ppl.run(input_data=None)


if __name__ == '__main__':
    unittest.main()
