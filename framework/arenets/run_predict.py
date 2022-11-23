from os.path import dirname, realpath, join

from arekit.common.experiment.data_type import DataType
from arekit.contrib.networks.enum_name_types import ModelNames

from framework.arenets.predict import predict_nn


if __name__ == '__main__':

    current_dir = dirname(realpath(__file__))
    nn_dir = join(current_dir, "../../_out/serialize-nn")

    for data_type in [DataType.Train, DataType.Test, DataType.Dev]:
        predict_nn(model_name=ModelNames.CNN, output_dir=nn_dir, data_type=data_type,
                   embedding_dir=nn_dir, samples_dir=nn_dir)
