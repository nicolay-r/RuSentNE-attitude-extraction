from os.path import dirname, realpath, join

from arekit.common.pipeline.base import BasePipeline
from framework.deeppavlov.states import BERT_CONFIG_PATH, BERT_DO_LOWERCASE, BERT_VOCAB_PATH, \
    BERT_FINETUNED2_MODEL_PATHDIR, BERT_FINETUNED_CKPT_PATH
from framework.deeppavlov.train import BertFinetunePipelineItem


if __name__ == '__main__':

    ppl = BasePipeline([
        BertFinetunePipelineItem(bert_config_file=BERT_CONFIG_PATH,
                                 model_checkpoint_path=BERT_FINETUNED_CKPT_PATH,
                                 vocab_filepath=BERT_VOCAB_PATH,
                                 do_lowercase=BERT_DO_LOWERCASE,
                                 max_seq_length=128,
                                 learning_rate=2e-5,
                                 save_path=BERT_FINETUNED2_MODEL_PATHDIR)
        ])

    current_dir = dirname(realpath(__file__))
    ppl.run(input_data=join(current_dir, "_out/serialize-bert/sample-train-0.tsv.gz"),
            params_dict={"epochs_count": 2, "batch_size": 6})
