import unittest
from os.path import dirname, realpath, join

from arekit.common.data.input.writers.tsv import TsvWriter
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
    output_nn_dir = join(output_dir, "serialize-nn")
    output_bert_dir = join(output_dir, "serialize-bert")

    def test_nn_tsv(self):
        serialize_nn(limit=1,
                     output_dir=self.output_nn_dir,
                     folding_type="fixed",
                     writer=TsvWriter(write_header=True),
                     labels_scaler=PosNegNeuRelationsLabelScaler(),
                     split_filepath="../data/split_fixed.txt")

    def test_nn_json(self):
        serialize_nn(limit=1,
                     output_dir=self.output_nn_dir,
                     folding_type="fixed",
                     writer=OpenNREJsonWriter(),
                     labels_scaler=PosNegNeuRelationsLabelScaler(),
                     split_filepath="../data/split_fixed.txt")

    def test_bert_json(self):
        serialize_bert(limit=1,
                       terms_per_context=50,
                       output_dir=self.output_bert_dir,
                       split_filepath="../data/split_fixed.txt",
                       writer=OpenNREJsonWriter(text_columns_type="bert"),
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
