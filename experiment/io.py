from arekit.common.data.input.writers.tsv import TsvWriter
from arekit.common.data.input.writers.base import BaseWriter
from arekit.contrib.utils.model_io.tf_networks import DefaultNetworkIOUtils

from writers.opennre_json import OpenNREJsonWriter


class InferIOUtils(DefaultNetworkIOUtils):

    def __init__(self, output_dir, exp_ctx):
        assert(isinstance(output_dir, str))
        super(InferIOUtils, self).__init__(exp_ctx=exp_ctx)
        self.__output_dir = output_dir

    def _get_experiment_sources_dir(self):
        return self.__output_dir


class CustomNetworkSerializationIO(DefaultNetworkIOUtils):

    def __init__(self, output_dir, exp_ctx, writer):
        assert(isinstance(output_dir, str))
        assert(isinstance(writer, BaseWriter) or writer is None)
        super(CustomNetworkSerializationIO, self).__init__(exp_ctx=exp_ctx)
        self.__output_dir = output_dir
        self.__writer = writer if writer is not None else TsvWriter(write_header=True)

    def _get_experiment_sources_dir(self):
        return self.__output_dir

    def create_samples_writer(self):
        return self.__writer

    def create_target_extension(self):
        if isinstance(self.__writer, TsvWriter):
            return ".tsv.gz"
        if isinstance(self.__writer, OpenNREJsonWriter):
            return ".json"


class CustomExperimentTrainIO(DefaultNetworkIOUtils):

    def __init__(self, source_dir, exp_ctx):
        assert(isinstance(source_dir, str))
        super(CustomExperimentTrainIO, self).__init__(exp_ctx=exp_ctx)
        self.__source_dir = source_dir

    def _get_experiment_sources_dir(self):
        return self.__source_dir
