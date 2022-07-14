from models.nn.serialize import serialize_nn


if __name__ == '__main__':
    serialize_nn(output_dir="_out", split_filepath="data/split_fixed.txt")
