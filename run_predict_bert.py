from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter

from labels.scaler import PosNegNeuRelationsLabelScaler
from models.bert.predict import BertInferencePipelineItem
from models.bert.states import BERT_CONFIG_PATH, BERT_FINETUNED_CKPT_PATH, BERT_VOCAB_PATH, BERT_DO_LOWERCASE, \
    BERT_CKPT_PATH, BERT_FINETUNED2_CKPT_PATH


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
