# OpenNRE framework

We adopt [OpenNRE](https://github.com/thunlp/OpenNRE) project.

This project might be utilized independently from the original one, i.e.
installed in a separated folder.

## Dependencies

Here, is a list of the related dependencies:

```
torch==1.6.0
transformers==3.4.0
numpy==1.19.5
pytest==5.3.2
scikit-learn==0.22.1
scipy==1.4.1
nltk>=3.6.4
```

The [complete pip packages lists](pip-freeze-list.txt) experiments were provided.

## Usage

### Data

You need to provide data in `JSONL` format.

### Training and Inferring

Please refer to a simple scripts `test_training.py` and `test_infer.py` for a greater details.