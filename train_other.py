from arekit.contrib.networks.enum_name_types import ModelNames

from run_train_nn import train_nn

# train_nn(output_dir="_out",
#          model_log_dir="_model",
#          model_name=ModelNames.PCNN,
#          split_source="data/split_fixed.txt")

train_nn(output_dir="_out",
         model_log_dir="_model",
         model_name=ModelNames.LSTM,
         split_source="data/split_fixed.txt")

train_nn(output_dir="_out",
         model_log_dir="_model",
         model_name=ModelNames.BiLSTM,
         split_source="data/split_fixed.txt")
