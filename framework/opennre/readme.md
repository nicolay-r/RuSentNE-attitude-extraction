# OpenNRE framework

We adopt [OpenNRE](https://github.com/thunlp/OpenNRE) project.

This project might be utilized independently from the original one, i.e.
installed in a separated folder.

## Dependencies

List of the related dependencies:

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

You need to provide data in `JSONL` format at [data folder](data).

By default, we provide a cropped by `100` entries:
* `nn-collection` (for CNN-based neural networks)
* `bert-collection` for bert-based models.

### List of the pretrained states

You need to provide the pre-trained state (if you will) at `ckpt`:

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

### Training and Inferring

> [test_training.py](test_training.py) 

> [test_infer.py](test_infer.py)

