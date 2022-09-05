from arekit.common.data.input.writers.opennre_json import OpenNREJsonWriter
from arekit.common.data.input.writers.tsv import TsvWriter
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper
from arekit.contrib.utils.bert.text_b_rus import BertTextBTemplates
from SentiNEREL.entity.formatter import CustomTypedEntitiesFormatter
from models.bert.serialize import CroppedBertSampleRowProvider, serialize_bert

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
            text_b_template=BertTextBTemplates.NLI.value,
            text_terms_mapper=BertDefaultStringTextTermsMapper(
                entity_formatter=CustomTypedEntitiesFormatter()
            )))


if __name__ == '__main__':

    do(TsvWriter(write_header=True))
    # do(OpenNREJsonWriter(text_columns=["text_a", "text_b"]))
