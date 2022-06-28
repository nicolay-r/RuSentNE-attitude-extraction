import os

from arekit.contrib.utils.model_io.tf_networks import DefaultNetworkIOUtils


class CustomExperimentSerializationIO(DefaultNetworkIOUtils):

    def __init__(self, output_dir, exp_ctx):
        assert(isinstance(output_dir, str))
        super(CustomExperimentSerializationIO, self).__init__(exp_ctx=exp_ctx)
        self.__output_dir = output_dir

    def __create_annot_input_target(self, doc_id, data_type):
        filename = "annot_input_d{doc_id}_{data_type}.txt".format(doc_id=doc_id, data_type=data_type.name)
        return os.path.join(self._get_target_dir(), filename)

    def _get_experiment_sources_dir(self):
        return self.__output_dir

    def create_opinion_collection_target(self, doc_id, data_type, check_existance=False):
        return self.__create_annot_input_target(doc_id=doc_id, data_type=data_type)


class CustomExperimentTrainIO(DefaultNetworkIOUtils):

    def __init__(self, source_dir, exp_ctx):
        assert(isinstance(source_dir, str))
        super(CustomExperimentTrainIO, self).__init__(exp_ctx=exp_ctx)
        self.__source_dir = source_dir

    def _get_experiment_sources_dir(self):
        return self.__source_dir
