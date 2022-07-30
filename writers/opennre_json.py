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
    def __optional_write_bag(bag, json_file):
        assert(isinstance(bag, dict) or bag is None)

        if bag is None:
            return

        bag_contents = bag[OpenNREJsonWriter.BAG_TAG]
        if len(bag_contents) == 0:
            return

        data_to_write = bag if len(bag_contents) > 1 else bag_contents[0]
        json.dump(data_to_write, json_file, separators=(",", ":"), ensure_ascii=False)
        json_file.write("\n")

    @staticmethod
    def __create_bag(row_id):
        return {
            OpenNREJsonWriter.BAG_TAG: [],
            "h": {"id": row_id + "sb"},
            "t": {"id": row_id + "tb"}
        }

    def save(self, storage, target):
        assert(isinstance(storage, BaseRowsStorage))
        assert(isinstance(target, str))

        df = storage.DataFrame
        df.sort_values(by=[const.ID], ascending=True)

        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding=self.__encoding) as json_file:

            bag = None

            for row_index, row in storage:

                s_ind = int(row["s_ind"])
                t_ind = int(row["t_ind"])

                json_row = {}
                json_row["id"] = row["id"]
                json_row["token"] = row["text_a"].split()
                json_row["h"] = {"pos": [s_ind, s_ind + 1], "id": str(row["id"] + "s")}
                json_row["t"] = {"pos": [t_ind, t_ind + 1], "id": str(row["id"] + "t")}

                if "label" in row:
                    json_row["relation"] = str(int(row["label"]))

                if bag is None or ("i0_" in json_row["id"] and len(bag) > 0):
                    self.__optional_write_bag(bag, json_file=json_file)
                    bag = self.__create_bag(row_id=row["id"])

                bag[self.BAG_TAG].append(json_row)

            self.__optional_write_bag(bag, json_file=json_file)

        logger.info("Saving completed!")
        logger.info(df.info())
