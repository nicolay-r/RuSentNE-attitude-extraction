from framework.arekit.serialize_nn import serialize_nn
from arekit.contrib.utils.data.writers.csv_pd import PandasCsvWriter

serialize_nn(output_dir="_out/serialize-nn",
             split_filepath="data/split_fixed.txt",
             writer=PandasCsvWriter(write_header=True))