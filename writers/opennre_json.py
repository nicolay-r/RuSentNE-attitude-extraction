import json
import os

from arekit.common.data import const
from arekit.common.data.input.writers.base import BaseWriter
from arekit.common.data.storages.base import BaseRowsStorage, logger


class OpenNREJsonWriter(BaseWriter):
    """ This is a bag-based writer for the samples.
    """

    BAG_TAG = "anno_relation_list"

    def __init__(self, encoding="utf-8"):
        assert(isinstance(encoding, str))
        self.__encoding = encoding

    @staticmethod
    def __write_bag(bag, json_file):
        json.dump(bag, json_file, separators=(",", ":"), ensure_ascii=False)
        json_file.write("\n")

    def save(self, storage, target):
        assert(isinstance(storage, BaseRowsStorage))
        assert(isinstance(target, str))

        df = storage.DataFrame
        df.sort_values(by=[const.ID], ascending=True)

        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding=self.__encoding) as json_file:

            for row_index, row in storage:

                s_ind = int(row["s_ind"])
                t_ind = int(row["t_ind"])
                sample_id = row["id"]
                bag_id = sample_id[0:sample_id.find('_i')]

                json_row = {}

                json_row["id"] = bag_id
                json_row["token"] = row["text_a"].split()
                json_row["h"] = {"pos": [s_ind, s_ind + 1], "id": str(bag_id + "s")}
                json_row["t"] = {"pos": [t_ind, t_ind + 1], "id": str(bag_id + "t")}
                json_row["relation"] = str(int(row["label"])) if "label" in row else "NA"

                self.__write_bag(json_row, json_file=json_file)

        logger.info("Saving completed!")
        logger.info(df.info())
