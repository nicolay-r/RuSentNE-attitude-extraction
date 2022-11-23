# AREnets

This is an [AREkit](https://github.com/nicolay-r/AREkit) `network` project contributional module.

## Train

> [run_train.py](run_train_nn.py)
 
Using the embedded `tensorflow`-based models.
The related AREkit module provides a 
[list of the supported models](https://github.com/nicolay-r/AREkit/tree/0.22.1-rc/arekit/contrib/networks#models-list),
dedicated for the sentiment relation extraction (`ModelNames` enum type).
Model training process, based on the SentiNEREL could be launched as follows:

> **NOTE:** considering a root project dir for this script

```python
from arekit.contrib.networks.enum_name_types import ModelNames
from framework.arenets.train import train_nn

train_nn(output_dir="_out/serialize-nn",
         model_log_dir="_model",
         model_name=ModelNames.AttEndsCNN,
         split_filepath="data/split_fixed.txt")
```

The latter produces the model at `_out/serialize_nn` with logging information at `_model` dir, and 
data split based on the `data/split_fixed.txt` file.


## Predict

> [run_predict.py](run_predict.py)

Belowe is an example which illustrates on how `CNN` model might be adopted as follows:

```python
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.common.experiment.data_type import DataType
from framework.arenets.predict import predict_nn

predict_nn(model_name=ModelNames.CNN, 
           output_dir="_out/serialize-nn", data_type=DataType.Test,
           embedding_dir="_out/serialize-nn", samples_dir="_out/serialize-nn")
```

The pretrained state will be automaticaly searched, and the latter
 provided via `model_io` at `predict_nn` [implementation](predict.py).
