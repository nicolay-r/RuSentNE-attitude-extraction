from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper

from entity.formatter import CustomEntitiesFormatter
from labels.formatter import PosNegNeuRelationsLabelFormatter
from labels.scaler import PosNegNeuRelationsLabelScaler
from models.bert.serialize import CroppedBertSampleRowProvider, bert_serialize_pipeline_item

if __name__ == '__main__':

    ppl = BasePipeline([
        bert_serialize_pipeline_item(
            terms_per_context=50,
            output_dir="_out",
            split_filepath="data/split_fixed.txt",
            name_provider=ExperimentNameProvider(name="serialize", suffix="bert"),
            sample_row_provider=CroppedBertSampleRowProvider(
                crop_window_size=50,
                label_scaler=PosNegNeuRelationsLabelScaler(),
                text_b_labels_fmt=PosNegNeuRelationsLabelFormatter(),
                text_terms_mapper=BertDefaultStringTextTermsMapper(
                    entity_formatter=CustomEntitiesFormatter(subject_fmt="#S", object_fmt="#O")
                )))
    ])

    ppl.run(input_data=None)
