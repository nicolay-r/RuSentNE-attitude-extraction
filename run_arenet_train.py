from arekit.contrib.networks.enum_name_types import ModelNames
from framework.arenets.train import train_nn


train_nn(output_dir="_out/serialize-nn",
         model_log_dir="_model",
         model_name=ModelNames.AttEndsCNN,
         split_filepath="data/split_fixed.txt")