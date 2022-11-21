import gzip
import json

import torch
from opennre.encoder import PCNNEncoder, CNNEncoder, BERTEncoder, BERTEntityEncoder
from opennre.framework import SentenceRE, SentenceRELoader


def create_framework(framework_name, model, train_path, test_path, val_path, ckpt, batch_size):
    if framework_name == "sentence":
        return SentenceRE(
            train_path=train_path,
            test_path=test_path,
            val_path=val_path,
            model=model,
            ckpt=ckpt,
            batch_size=batch_size,
            max_epoch=41,
            lr=0.1,
            weight_decay=1e-5,
            opt="sgd")


def create_framework_eval_loader(framework_name, path, rel2id, tokenizer, batch_size):
    if framework_name == "sentence":
        return SentenceRELoader(path=path, rel2id=rel2id, tokenizer=tokenizer,
                                batch_size=batch_size, shuffle=False)


def load_sentence_encoder(model_name, word2vec, word2id, dropout):
    if model_name == "pcnn":
        return PCNNEncoder(
            token2id=word2id,
            max_length=50,
            word_size=word2vec.shape[1],
            position_size=5,
            hidden_size=2000,
            blank_padding=True,
            kernel_size=3,
            padding_size=1,
            word2vec=word2vec,
            dropout=dropout)
    if model_name == "cnn":
        return CNNEncoder(
            token2id=word2id,
            max_length=50,
            word_size=word2vec.shape[1],
            position_size=5,
            hidden_size=2000,
            blank_padding=True,
            kernel_size=3,
            padding_size=1,
            word2vec=word2vec,
            dropout=dropout)


def load_bert_sentence_encoder(pooler, max_length, pretrain_path, mask_entity):
    if pooler == 'entity':
        return BERTEntityEncoder(
            max_length=max_length,
            pretrain_path=pretrain_path,
            mask_entity=mask_entity
        )
    elif pooler == 'cls':
        return BERTEncoder(
            max_length=max_length,
            pretrain_path=pretrain_path,
            mask_entity=mask_entity
        )
    else:
        raise NotImplementedError


def write_unique_predict(output_file_gzip, rels_count, res_iter, sep="\t"):
    written = set()
    with gzip.open(output_file_gzip, "wb") as csv_output:

        # Print header.
        csv_output.write(sep.join(["id"] + [str(i) for i in range(rels_count)]).encode())
        csv_output.write("\n".encode())

        for json_row_result in res_iter:
            assert(isinstance(json_row_result, dict))

            # Pick relation.
            relation = json_row_result["relation"]

            # To vector.
            vector = ["0"] * rels_count
            vector[relation] = "1"

            # Output optional.
            sample_id = json_row_result["id"]
            if sample_id not in written:
                csv_output.write("{id}{sep}{v}\n".format(id=sample_id, sep=sep, v=sep.join(vector)).encode())

            # Provide as written.
            written.add(sample_id)


def extract_ids(data_file):
    with open(data_file) as input_file:
        for line_str in input_file.readlines():
            data = json.loads(line_str)
            yield data["id_orig"]


def iter_results(parallel_model, eval_loader, data_ids):
    # Batch size may vary, so we then adopt a
    # separated label for line index tracking.
    l_ind = 0
    with torch.no_grad():
        for iter, data in enumerate(eval_loader):
            if torch.cuda.is_available():
                for i in range(len(data)):
                    try:
                        data[i] = data[i].cuda()
                    except:
                        pass

            args = data[1:]
            logits = parallel_model(*args)
            score, pred = logits.max(-1)  # (B)

            # Save result
            batch_size = pred.size(0)
            for i in range(batch_size):
                yield {"id": data_ids[l_ind], "relation": pred[i].item()}
                l_ind += 1
