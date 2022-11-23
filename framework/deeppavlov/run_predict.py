from arekit.common.experiment.data_type import DataType
from framework.deeppavlov.predict_pipeline import predict_bert
from framework.deeppavlov.states import BERT_FINETUNED2_CKPT_PATH


if __name__ == '__main__':
    for data_type in [DataType.Train, DataType.Test, DataType.Dev]:
        predict_bert(full_model_name="bert-ra-rsr-ft",
                     ckpt_path=BERT_FINETUNED2_CKPT_PATH,
                     data_type=data_type)
