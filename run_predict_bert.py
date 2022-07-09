from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.experiment.api.ctx_serialization import ExperimentSerializationContext
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter

from labels.scaler import PosNegNeuRelationsLabelScaler
from models.bert.predict import BertInferencePipelineItem
from models.bert.states import BERT_CONFIG_PATH, BERT_FINETUNED_CKPT_PATH, BERT_VOCAB_PATH, BERT_DO_LOWERCASE, \
    BERT_CKPT_PATH, BERT_FINETUNED2_CKPT_PATH


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


if __name__ == '__main__':

    ppl = BasePipeline([
        BertInferencePipelineItem(
            bert_config_file=BERT_CONFIG_PATH,
            do_lowercase=BERT_DO_LOWERCASE,
            labels_scaler=PosNegNeuRelationsLabelScaler(),
            predict_writer=TsvPredictWriter(),
            max_seq_length=128,
            model_checkpoint_path=BERT_FINETUNED_CKPT_PATH,
            vocab_filepath=BERT_VOCAB_PATH)
        ])

    ppl.run(input_data={"predict_dir": "_out/serialize-bert_rsr_1l",
                        "samples_filepath": "_out/serialize-bert_1l/sample-test-0.tsv.gz"},
            params_dict={"full_model_name": "bert_rsr"})
