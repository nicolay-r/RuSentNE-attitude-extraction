from models.nn.serialize import serialize_nn
from writers.opennre_json import OpenNREJsonWriter

if __name__ == '__main__':
    serialize_nn(output_dir="_out", split_filepath="data/split_fixed.txt",
                 writer=OpenNREJsonWriter())
