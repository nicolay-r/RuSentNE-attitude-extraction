from arekit.contrib.networks.enum_name_types import ModelNames
from models.nn.predict import predict_nn


if __name__ == '__main__':
    predict_nn(extra_name_suffix="nn", model_name=ModelNames.CNN, output_dir="_out")
