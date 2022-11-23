from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter

from SentiNEREL.labels.scaler import PosNegNeuRelationsLabelScaler
from framework.deeppavlov.predict_pipeline_item import BertInferencePipelineItem
from framework.deeppavlov.states import BERT_CONFIG_PATH, BERT_FINETUNED_CKPT_PATH, BERT_VOCAB_PATH, BERT_DO_LOWERCASE


def predict_bert(max_seq_length=128, bert_config=BERT_CONFIG_PATH, do_lowercase=BERT_DO_LOWERCASE,
                 ckpt_path=BERT_FINETUNED_CKPT_PATH, vocab_filepath=BERT_VOCAB_PATH,
                 predict_dir="_out/serialize-bert", samples_dir="_out/serialize-bert",
                 full_model_name="bert", data_type=DataType.Test):
    """ Wrapping predict pipeline item into a complete pipeline.
    """

    ppl = BasePipeline([
        BertInferencePipelineItem(
            bert_config_file=bert_config,
            do_lowercase=do_lowercase,
            labels_scaler=PosNegNeuRelationsLabelScaler(),
            predict_writer=TsvPredictWriter(),
            max_seq_length=max_seq_length,
            model_checkpoint_path=ckpt_path,
            vocab_filepath=vocab_filepath,
            data_type=data_type)
    ])

    ppl.run(input_data={"predict_dir": predict_dir,
                        "samples_dir": samples_dir},
            params_dict={"full_model_name": full_model_name})
