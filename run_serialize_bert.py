from arekit.common.data.input.writers.tsv import TsvWriter
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper
from arekit.contrib.utils.bert.text_b_rus import BertTextBTemplates
from entity.formatter import CustomEntitiesFormatter
from models.bert.serialize import CroppedBertSampleRowProvider, serialize_bert
from writers.opennre_json import OpenNREJsonWriter

from SentiNEREL.labels.formatter import PosNegNeuRelationsLabelFormatter
from SentiNEREL.labels.scaler import PosNegNeuRelationsLabelScaler


def do(writer):
    serialize_bert(
        terms_per_context=50,
        output_dir="_out/serialize-bert/",
        split_filepath="data/split_fixed.txt",
        writer=writer,
        sample_row_provider=CroppedBertSampleRowProvider(
            crop_window_size=50,
            label_scaler=PosNegNeuRelationsLabelScaler(),
            text_b_labels_fmt=PosNegNeuRelationsLabelFormatter(),
            text_b_template=BertTextBTemplates.NLI.value,
            text_terms_mapper=BertDefaultStringTextTermsMapper(
                entity_formatter=CustomEntitiesFormatter(subject_fmt="#S", object_fmt="#O")
            )))


if __name__ == '__main__':

    do(TsvWriter(write_header=True))
    # do(OpenNREJsonWriter("bert"))
