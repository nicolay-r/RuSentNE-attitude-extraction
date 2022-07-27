from arekit.contrib.utils.model_io.tf_networks import DefaultNetworkIOUtils


class InferIOUtils(DefaultNetworkIOUtils):

    def __init__(self, output_dir, exp_ctx):
        assert(isinstance(output_dir, str))
        super(InferIOUtils, self).__init__(exp_ctx=exp_ctx)
        self.__output_dir = output_dir

    def _get_experiment_sources_dir(self):
        return self.__output_dir


class CustomExperimentSerializationIO(DefaultNetworkIOUtils):

    def __init__(self, output_dir, exp_ctx):
        assert(isinstance(output_dir, str))
        super(CustomExperimentSerializationIO, self).__init__(exp_ctx=exp_ctx)
        self.__output_dir = output_dir

    def _get_experiment_sources_dir(self):
        return self.__output_dir


class CustomExperimentTrainIO(DefaultNetworkIOUtils):

    def __init__(self, source_dir, exp_ctx):
        assert(isinstance(source_dir, str))
        super(CustomExperimentTrainIO, self).__init__(exp_ctx=exp_ctx)
        self.__source_dir = source_dir

    def _get_experiment_sources_dir(self):
        return self.__source_dir
