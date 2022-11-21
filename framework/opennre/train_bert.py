import json
import os

import torch
from opennre.framework import SentenceRE
from opennre.model import SoftmaxNN

from framework.opennre.utils import load_bert_sentence_encoder


def run_finetunning_bert(data_root_path, pretrain_path, ckpt_source, ckpt_target,
                         pooler, max_length=128, batch_size=6, max_epoch=4, lr=5e-6):
    """ Finetunning already pre-trained state.
    """

    rel2id = json.load(open(os.path.join(data_root_path, "rel2id.json")))
    train_path = os.path.join(data_root_path, "sample-train-0.json")
    test_path = os.path.join(data_root_path, "sample-test-0.json")
    val_path = train_path

    sentence_encoder = load_bert_sentence_encoder(
        pooler=pooler, mask_entity=True, max_length=max_length, pretrain_path=pretrain_path)

    # Define the model output and load already provided checkpoint
    model = SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
    model.load_state_dict(torch.load(ckpt_source, map_location='cpu')['state_dict'])

    # Define the whole training framework
    framework = SentenceRE(
        train_path=train_path,
        test_path=test_path if os.path.exists(test_path) else None,
        val_path=val_path,
        model=model,
        ckpt=ckpt_target,
        batch_size=batch_size,
        max_epoch=max_epoch,
        lr=lr,
        opt='adamw')

    # Train the model
    framework.train_model('micro_f1')


def run_training_bert(pretrain_path, data_root_path, pooler, ckpt_target=None,
                      max_length=128, batch_size=6, max_epoch=4, lr=1e-5):
    """ Training BERT from the original state.
    """

    rel2id = json.load(open(os.path.join(data_root_path, "rel2id.json")))
    train_path = os.path.join(data_root_path, "sample-train-0.json")
    test_path = os.path.join(data_root_path, "sample-test-0.json")
    val_path = train_path

    # Define the sentence encoder
    sentence_encoder = load_bert_sentence_encoder(
        pooler=pooler, mask_entity=True, max_length=max_length, pretrain_path=pretrain_path)

    # Define the model
    model = SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

    # Define the whole training framework
    framework = SentenceRE(
        train_path=train_path,
        test_path=test_path if os.path.exists(test_path) else None,
        val_path=val_path,
        model=model,
        ckpt=ckpt_target,
        batch_size=batch_size,
        max_epoch=max_epoch,
        lr=lr,
        opt='adamw')

    # Train the model
    framework.train_model('micro_f1')
