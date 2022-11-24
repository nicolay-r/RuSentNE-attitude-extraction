# SentiNEREL collection Serialization

This README represents a list of scripts tutorial required to perform SentiNEREL collection serialization.

> **NOTE:** Run from the main project folder.

## For neural networks
```python
from framework.arekit.serialize_nn import serialize_nn
from arekit.contrib.utils.data.writers.csv_pd import PandasCsvWriter

serialize_nn(output_dir="_out/serialize-nn", 
             split_filepath="data/split_fixed.txt", 
             writer=PandasCsvWriter(write_header=True))
```

## For BERT model
```python
from SentiNEREL.entity.formatter import CustomTypedEntitiesFormatter
from SentiNEREL.labels.scaler import PosNegNeuRelationsLabelScaler
from arekit.contrib.bert.terms.mapper import BertDefaultStringTextTermsMapper
from arekit.contrib.utils.bert.text_b_rus import BertTextBTemplates
from arekit.contrib.utils.data.writers.csv_pd import PandasCsvWriter
from framework.arekit.serialize_bert import CroppedBertSampleRowProvider, serialize_bert

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
