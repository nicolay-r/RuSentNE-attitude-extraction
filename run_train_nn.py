from arekit.contrib.networks.enum_name_types import ModelNames

from models.nn.train import train_nn

if __name__ == '__main__':

    train_nn(output_dir="_out/serialize-nn",
             model_log_dir="_model",
             model_name=ModelNames.CNN,
             split_filepath="data/split_fixed.txt")

    train_nn(output_dir="_out/serialize-nn",
             model_log_dir="_model",
             model_name=ModelNames.PCNN,
             split_filepath="data/split_fixed.txt")

    train_nn(output_dir="_out/serialize-nn",
             model_log_dir="_model",
             model_name=ModelNames.AttEndsCNN,
             split_filepath="data/split_fixed.txt")

    train_nn(output_dir="_out/serialize-nn",
             model_log_dir="_model",
             model_name=ModelNames.AttEndsPCNN,
             split_filepath="data/split_fixed.txt")
