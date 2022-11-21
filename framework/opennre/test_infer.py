from framework.opennre.infer_bert import infer_bert

if __name__ == '__main__':

    infer_bert(pretrain_path="DeepPavlov/rubert-base-cased",
               collection="rsne8",
               dtype="dev",
               root_path="./data/bert-collection",
               ckpt_dir="ckpt-nov",
               pooler="cls")