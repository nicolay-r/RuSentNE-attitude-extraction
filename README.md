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
* [Download Finetuned Models](#download-finetuned-models)
* [Serialize SentiNEREL](#serialize-collection)
* [Frameworks](#frameworks)
    * [AREnets](framework/arenets) directory
    * [OpenNRE](framework/opennre) directory
    * [DeepPavlov](framework/deeppavlov) directory
    * [Hitachi-graph-based](framework/hitachi_graph) directory
* [Pretrained States](#pretrained-states)
* [Sponsors](#sponsors)

## Installation

```python
pip install -r dependencies.txt
```

## Serialize Collection

For conventional neural networks:
```python
from framework.arenets.serialize import serialize_nn
from arekit.contrib.utils.data.writers.csv_pd import PandasCsvWriter

serialize_nn(output_dir="_out/serialize-nn", 
             split_filepath="data/split_fixed.txt", 
             writer=PandasCsvWriter(write_header=True))
```

For `BERT` model:
```python
from SentiNEREL.entity.formatter import CustomTypedEntitiesFormatter
from SentiNEREL.labels.scaler import PosNegNeuRelationsLabelScaler
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper
from arekit.contrib.utils.bert.text_b_rus import BertTextBTemplates
from arekit.contrib.utils.data.writers.csv_pd import PandasCsvWriter
from framework.deeppavlov.serialize import CroppedBertSampleRowProvider, serialize_bert

def do(writer):
    serialize_bert(
        terms_per_context=50,
        output_dir="_out/serialize-bert/",
        split_filepath="data/split_fixed.txt",
        writer=PandasCsvWriter(write_header=True),
        sample_row_provider=CroppedBertSampleRowProvider(
            crop_window_size=50,
            label_scaler=PosNegNeuRelationsLabelScaler(),
            text_b_template=BertTextBTemplates.NLI.value,
            text_terms_mapper=BertDefaultStringTextTermsMapper(
                entity_formatter=CustomTypedEntitiesFormatter()
            )))

do(PandasCsvWriter(write_header=True))        # CSV-based output
# do(OpenNREJsonWriter(text_columns=["text_a", "text_b"]))  # JSONL-based output
```

[Back to Top](#contents)

# Frameworks
   
* [opennre](framework/opennre/readme.md) -- based on OpenNRE toolkit (BERT-based models).
* [arenets](framework/arenets/readme.md) -- based on AREkit, tensorflow-based module 
for neural network training/finetunning/inferring.
* [deeppavlov](framework/deeppavlov/readme.md) `[legacy]` -- based on DeepPavlov framework (BERT-based models).
* [hittachi-graph-based](framework/hitachi_graph/readme.md) -- provides implementation of the graph-based 
approaches over transformers.

[Back to Top](#contents)

# Pretrained states
List of the `OpenNRE` pretrained, BERT-based models:
* [ra4_DeepPavlov-rubert-base-cased_cls.pth](https://disk.yandex.ru/d/fuGqPNBXPigttQ)
   * RuAttitudes (4 ep.), with `cls` based pooling scheme;
* [ra4_DeepPavlov-rubert-base-cased_entity.pth](https://disk.yandex.ru/d/ep_O-c1YVgu3Dw)
   * RuAttitudes (4 ep.), with `entity` based pooling scheme;
* [ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth](https://disk.yandex.ru/d/OwA6h5BioA9LOw)
   * RuAttitudes (4 ep.) + RuSentRel (1 ep.), with `cls` pooling scheme;
* [ra4-rsr1_DeepPavlov-rubert-base-cased_entity.pth](https://disk.yandex.ru/d/_SoRgM5pLVgVoQ)
   * RuAttitudes (4 ep.),+ RuSentRel (1 ep.), with `entity` pooling scheme;
* [ra4-rsr1-rsne4_DeepPavlov-rubert-base-cased_cls.pth](https://disk.yandex.ru/d/Ae09HxlKoOodHw) 
   * RuAttitudes (4 ep.) + RuSentRel (1 ep.) + SentiNEREL-train (4 ep.), with `cls` based pooling scheme;
* [ra4-rsr1-rsne4_DeepPavlov-rubert-base-cased_entity.pth](https://disk.yandex.ru/d/5YLbxDBR5EsJvg) 
   * RuAttitudes (4 ep.) + RuSentRel (1 ep.) + SentiNEREL-train (4 ep.), with `entity` based pooling scheme;

[Back to Top](#contents)

### Sponsors

<p align="left">
    <img src="data/images/logo_msu.png"/>
</p>
