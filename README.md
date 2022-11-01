## RuSentNE Sentiment Attitude Extraction Studies

![](https://img.shields.io/badge/Python-3.6-brightgreen.svg)
![](https://img.shields.io/badge/AREkit-0.23.0-orange.svg)

This repository represents studies related to sentiment attitude extraction, provided for 
sentiment relations of the [NEREL-based dataset](https://github.com/nerel-ds/nerel), dubbed as **SentiNEREL**.

The following spreadsheet represents ML-models benchmark evaluation results
obtained for the sentiment attitude relation extraction:

> [Leaderboard Google Spreadsheet](https://docs.google.com/spreadsheets/d/1o4VVZZNraO_-dr-WnGU8LM2aEjTp8KjZhFmTab5e5DM/edit?usp=sharing)

Powered by [AREkit-0.23.0](https://github.com/nicolay-r/AREkit) framework, based on the tutorial:
[Binding a custom annotated collection for Relation Extraction](https://nicolay-r.github.io/blog/articles/2022-08/arekit-collection-bind).

## Contents

* [Installation](#installation)
* [Serialize SentiNEREL](#serialize-collection)
* [Training](#training)
    * [CNN/RNN-based models](#neural-networks)

## Installation

```python
pip install -r dependencies.txt
```

## Serialize Collection

For conventional neural networks:
```python
from models.nn.serialize import serialize_nn

serialize_nn(output_dir="_out/serialize-nn", 
             split_filepath="data/split_fixed.txt", 
             writer=TsvWriter(write_header=True))
```

For `BERT` model:
```python
from models.bert.serialize import CroppedBertSampleRowProvider, serialize_bert

serialize_bert(
    terms_per_context=50,
    output_dir="_out/serialize-bert/",
    split_filepath="data/split_fixed.txt",
    writer=TsvWriter(),
    sample_row_provider=CroppedBertSampleRowProvider(
        crop_window_size=50,
        label_scaler=PosNegNeuRelationsLabelScaler(),
        text_b_template=BertTextBTemplates.NLI.value,
        text_terms_mapper=BertDefaultStringTextTermsMapper(
            entity_formatter=CustomTypedEntitiesFormatter()
        )))
```

[Back to Top](#contents)

## Training 

### Neural Networks

Using the embedded `tensorflow`-based models.
The related AREkit module provides a 
[list of the supported models](https://github.com/nicolay-r/AREkit/tree/0.22.1-rc/arekit/contrib/networks#models-list),
dedicated for the sentiment relation extraction (`ModelNames` enum type).
Model training process, based on the SentiNEREL could be launched as follows:

```python
from models.nn.train import train_nn

train_nn(output_dir="_out/serialize-nn",
         model_log_dir="_model",
         model_name=ModelNames.AttEndsCNN,
         split_filepath="data/split_fixed.txt")
```

The latter produces the model at `_out/serialize_nn` with logging information at `_model` dir, and 
data split based on the `data/split_fixed.txt` file.

[Back to Top](#contents)

### Sponsors

<p align="left">
    <img src="data/images/logo_msu.png"/>
</p>
