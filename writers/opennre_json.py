import json
import os

from arekit.common.data import const
from arekit.common.data.input.writers.base import BaseWriter
from arekit.common.data.storages.base import BaseRowsStorage, logger


class OpenNREJsonWriter(BaseWriter):

    def __init__(self, encoding="utf-8"):
        assert(isinstance(encoding, str))
        self.__encoding = encoding

    @staticmethod
    def __write(data, json_file):
        json.dump(data, json_file, separators=(",", ":"), ensure_ascii=False)
        json_file.write("\n")

    def save(self, storage, target):
        assert(isinstance(storage, BaseRowsStorage))
        assert(isinstance(target, str))

        df = storage.DataFrame
        df.sort_values(by=[const.ID], ascending=True)

        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding=self.__encoding) as json_file:

            bag = []

            for row_index, row in storage:

                json_row = {}
                json_row["token"] = row["text_a"].split()
                s_ind = int(row["s_ind"])
                json_row["h"] = {"pos": [s_ind, s_ind + 1]}
                t_ind = int(row["t_ind"])
                json_row["t"] = {"pos": [t_ind, t_ind + 1]}

                if "label" in row:
                    json_row["relation"] = int(row["label"])

                json_row["id"] = row["id"]

                if "i0_" in json_row["id"] and len(bag) > 0:
                    self.__write(data=bag, json_file=json_file)
                    bag = []

                bag.append(json_row)

            if len(bag) > 0:
                self.__write(data=bag, json_file=json_file)

        logger.info("Saving completed!")
        logger.info(df.info())
