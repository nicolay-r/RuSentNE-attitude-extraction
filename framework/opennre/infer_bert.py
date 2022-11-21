import json
import os

import torch
from opennre.framework import SentenceRELoader
from opennre.model import SoftmaxNN
from tqdm import tqdm

from framework.opennre.utils import load_bert_sentence_encoder, iter_results, extract_ids, write_unique_predict


def infer_bert(pretrain_path, root_path, collection, ckpt_source=None, dtype="test", pooler='cls',
               ckpt_dir="ckpt", batch_size=6, max_length=128, mask_entity=True):
    """
    This is a main and core method for inference based on OpenNRE framework.
    """

    if ckpt_source is None:
        ckpt_source = os.path.join(ckpt_dir, '{col}_{pretrain_path}_{pooler}.pth.tar'.format(
            col=collection, pretrain_path=pretrain_path.replace('/', '-'), pooler=pooler))
        print(ckpt_source)

    rel2id = json.load(open(os.path.join(root_path, "rel2id.json")))
    test_data_file = os.path.join(root_path, "sample-{dtype}-0.json".format(dtype=dtype))
    output_file = os.path.join(root_path, "predict-{col}-{model}-{pooler}-{dtype}.tsv.gz".format(
        col=collection, model=pretrain_path.replace("/", '-'), pooler=pooler, dtype=dtype))

    # Download word2vec models.
    sentence_encoder = load_bert_sentence_encoder(
        pooler=pooler, mask_entity=mask_entity, max_length=max_length, pretrain_path=pretrain_path)

    # Load weights by the provided checkpoint.
    model = SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
    model.load_state_dict(torch.load(ckpt_source)['state_dict'])

    eval_loader = SentenceRELoader(test_data_file,
                                   model.rel2id,
                                   model.sentence_encoder.tokenize,
                                   batch_size,
                                   False)

    it_results = iter_results(parallel_model=torch.nn.DataParallel(model),
                              data_ids=list(extract_ids(test_data_file)),
                              eval_loader=eval_loader)

    write_unique_predict(output_file_gzip=output_file,
                         rels_count=len(rel2id),
                         res_iter=tqdm(it_results, desc=ckpt_source + " [{}]".format(dtype)))
