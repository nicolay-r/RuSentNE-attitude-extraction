from framework.opennre.train_bert import run_training_bert, run_finetunning_bert


def do_rsne_ft(pooler="entity", epochs=2):
    """ Training example.
    """
    model_name = "DeepPavlov/rubert-base-cased"
    ckpt_ra = './ckpt/{}.pth.tar'.format('{}_{}_{}'.format("rsne2.1", model_name.replace("/", "-"), pooler))
    run_training_bert(pretrain_path=model_name, pooler=pooler,
                      data_root_path="./data/bert-collection",
                      ckpt_target=ckpt_ra,
                      batch_size=16,
                      lr=1e-5,
                      max_epoch=epochs)


def do_ft(epochs=2, pooler="cls"):
    """ Fine-tunning example.
    """
    model_name = "DeepPavlov/rubert-base-cased"
    ckpt = './ckpt/{}.pth.tar'.format('{}_{}_{}'.format("rsne6", model_name.replace("/", "-"), pooler))
    ckpt_to = './ckpt/{}.pth.tar'.format('{}_{}_{}'.format("rsne8", model_name.replace("/", "-"), pooler))
    run_finetunning_bert(pretrain_path=model_name, pooler=pooler,
                         data_root_path="./data/bert-collection",
                         ckpt_source=ckpt,
                         ckpt_target=ckpt_to,
                         max_epoch=epochs,
                         batch_size=16)


# Use this code for the fine-tunning
do_ft()

