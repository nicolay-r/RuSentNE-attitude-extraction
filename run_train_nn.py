from arekit.contrib.networks.enum_name_types import ModelNames

from models.nn.train import train_nn

if __name__ == '__main__':

    train_nn(output_dir="_out",
             model_log_dir="_model",
             model_name=ModelNames.AttSelfPZhouBiLSTM,
             split_filepath="data/split_fixed.txt")
