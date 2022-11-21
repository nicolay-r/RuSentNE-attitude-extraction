import collections
import sys, json
import os
import random
import numpy as np
import torch
from opennre.model import SoftmaxNN

from framework.opennre.utils import load_sentence_encoder, create_framework


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def vocab2json(vocab):
    assert(isinstance(vocab, collections.Iterable))
    d = {}
    for word, index in vocab:
        d[word] = int(index)
    return d


def run_training_cnn(model_name, root_path, dataset="collection", framework_name="sentence", batch_size=40):

    # Some basic settings
    sys.path.append(root_path)
    if not os.path.exists('ckpt'):
        os.mkdir('ckpt')

    set_seed(42)

    # Setup paths.
    rel2id_file = os.path.join(root_path, "rel2id.json")
    vocab_file = os.path.join(root_path, "vocab-0.txt.npz")
    word2vec_file = os.path.join(root_path, 'term_embedding-0.npz')

    # Loading resources.
    word2id = vocab2json(np.load(vocab_file)["arr_0"])
    word2vec = np.load(word2vec_file)["arr_0"]
    rel2id = json.load(open(rel2id_file))

    sentence_encoder = load_sentence_encoder(model_name=model_name,
                                             word2id=word2id,
                                             word2vec=word2vec,
                                             dropout=0.1)

    model = SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

    test_path = os.path.join(root_path, "sample-test-0.json")

    # Define the whole training framework
    framework = create_framework(
        model=model,
        batch_size=batch_size,
        framework_name=framework_name,
        train_path=os.path.join(root_path, "sample-train-0.json"),
        test_path=test_path if os.path.exists(test_path) else None,
        val_path=os.path.join(root_path, "sample-train-0.json"),
        ckpt='ckpt/{dataset}-{model}.pth.tar'.format(dataset=dataset, model=model_name))

    framework.train_model('micro_f1')
