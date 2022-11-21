import os
import tarfile
from arekit.common import utils

from framework.deeppavlov.states import BERT_PRETRAINED_MODEL_TAR, BERT_FINETUNED_MODEL_TAR


def download_data():
    root_dir = "_model"

    data = {
        BERT_PRETRAINED_MODEL_TAR: "https://www.dropbox.com/s/cr6nejxjiqbyd5o/ra-20-srubert-large-neut-nli-pretrained-3l.tar.gz?dl=1",
        BERT_FINETUNED_MODEL_TAR: "https://www.dropbox.com/s/g73osmwyrqtr2at/ra-20-srubert-large-neut-nli-pretrained-3l-finetuned.tar.gz?dl=1"
    }

    # Perform downloading ...
    for local_name, url_link in data.items():
        print("Downloading: {}".format(local_name))
        utils.download(dest_file_path=os.path.join(root_dir, local_name),
                       source_url=url_link)

    # Extracting tar files ...
    for local_name in data.keys():
        dest_filepath = os.path.join(root_dir, local_name)
        if not os.path.exists(dest_filepath):
            continue
        if not tarfile.is_tarfile(dest_filepath):
            continue
        with tarfile.open(dest_filepath) as f:
            target = os.path.dirname(dest_filepath)
            f.extractall(path=target)

        # Remove .tar file
        os.remove(dest_filepath)


if __name__ == '__main__':
    download_data()
