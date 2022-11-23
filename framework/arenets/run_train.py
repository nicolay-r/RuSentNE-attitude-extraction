from os.path import dirname, realpath, join
from arekit.contrib.networks.enum_name_types import ModelNames
from framework.arenets.train import train_nn


if __name__ == '__main__':

    current_dir = dirname(realpath(__file__))

    # Run CNN model training,
    # based on the SentiNEREL fixed splitting.
    train_nn(output_dir=join(current_dir, "../../_out/serialize-nn"),
             model_log_dir=join(current_dir, "../../_model"),
             model_name=ModelNames.CNN,
             split_filepath=join(current_dir, "../../data/split_fixed.txt"))
