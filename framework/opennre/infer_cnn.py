import json
import os
import numpy as np
import torch
from opennre.model import SoftmaxNN
from tqdm import tqdm

from framework.opennre.train_cnn import vocab2json
from framework.opennre.utils import load_sentence_encoder, create_framework_eval_loader, \
    iter_results, extract_ids, write_unique_predict


def run_infer_cnn(root_path, encoder_name="cnn", dtype="test", framework_name="sentence",
                  batch_size=20, ckpt_dir="ckpt"):
    """
    This is a main and core method for inference based on OpenNRE framework.
    """

    ckpt = os.path.join(ckpt_dir, 'collection-{encoder_name}.pth.tar'.format(encoder_name=encoder_name))

    vocab_file = os.path.join(root_path, "vocab-0.txt.npz")
    rel2id_file = os.path.join(root_path, "rel2id.json")
    word2vec_file = os.path.join(root_path, "term_embedding-0.npz")
    test_data_file = os.path.join(root_path, "sample-{dtype}-0.json".format(dtype=dtype))
    output_file = os.path.join(root_path, "predict-opennre-{model}-{dtype}.tsv.gz".format(
        model=encoder_name, dtype=dtype))
    rel2id = json.load(open(rel2id_file))

    # Download word2vec models.
    word2id = vocab2json(np.load(vocab_file)["arr_0"])
    word2vec = np.load(word2vec_file)["arr_0"]
    sentence_encoder = load_sentence_encoder(
        model_name=encoder_name, word2id=word2id, word2vec=word2vec, dropout=0)

    model = SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

    model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])

    eval_loader = create_framework_eval_loader(framework_name=framework_name,
                                               path=test_data_file,
                                               rel2id=model.rel2id,
                                               tokenizer=model.sentence_encoder.tokenize,
                                               batch_size=batch_size)

    it_results = iter_results(parallel_model=torch.nn.DataParallel(model),
                              data_ids=list(extract_ids(test_data_file)),
                              eval_loader=eval_loader)

    write_unique_predict(output_file_gzip=output_file,
                         rels_count=len(rel2id),
                         res_iter=tqdm(it_results, desc=ckpt + " [{}]".format(dtype)))
