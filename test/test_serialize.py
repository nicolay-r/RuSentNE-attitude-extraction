import unittest
from os.path import dirname, realpath, join

from arekit.common.data.input.writers.tsv import TsvWriter
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper

from entity.formatter import CustomEntitiesFormatter
from labels.formatter import PosNegNeuRelationsLabelFormatter
from labels.scaler import PosNegNeuRelationsLabelScaler
from models.bert.serialize import CroppedBertSampleRowProvider, serialize_bert
from models.nn.serialize import serialize_nn
from writers.opennre_json import OpenNREJsonWriter


class TestSerialize(unittest.TestCase):

    current_dir = dirname(realpath(__file__))
    output_dir = join(current_dir, "_out")

    def test_nn(self):

        serialize_nn(limit=1,
                     output_dir=self.output_dir,
                     folding_type="fixed",
                     writer=TsvWriter(),
                     split_filepath="../data/split_fixed.txt")

    def test_nn_json(self):
        serialize_nn(limit=1,
                     output_dir=self.output_dir,
                     folding_type="fixed",
                     writer=OpenNREJsonWriter(),
                     split_filepath="../data/split_fixed.txt")

    def test_bert(self):

        serialize_bert(limit=1,
                       terms_per_context=50,
                       output_dir=self.output_dir,
                       split_filepath="../data/split_fixed.txt",
                       name_provider=ExperimentNameProvider(name="serialize", suffix="bert"),
                       folding_type="fixed",
                       sample_row_provider=CroppedBertSampleRowProvider(
                           crop_window_size=50,
                           label_scaler=PosNegNeuRelationsLabelScaler(),
                           text_b_labels_fmt=PosNegNeuRelationsLabelFormatter(),
                           text_terms_mapper=BertDefaultStringTextTermsMapper(
                               entity_formatter=CustomEntitiesFormatter(subject_fmt="#S", object_fmt="#O")
                           )))


if __name__ == '__main__':
    unittest.main()
