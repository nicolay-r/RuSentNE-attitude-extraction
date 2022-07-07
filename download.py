import os
import tarfile
from arekit.common import utils


def download_data():
    root_dir = "_model"

    BERT_PRETRAINED_MODEL_TAR = "ra-20-srubert-large-neut-nli-pretrained-3l.tar.gz"
    BERT_FINETUNED_MODEL_TAR = "ra-20-srubert-large-neut-nli-pretrained-3l-finetuned.tar.gz"

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
        if not os.path.exists(local_name):
            continue
        if not tarfile.is_tarfile(local_name):
            continue
        with tarfile.open(local_name) as f:
            target = os.path.dirname(local_name)
            f.extractall(path=target)

        # Remove .tar file
        os.remove(local_name)


if __name__ == '__main__':
    download_data()
