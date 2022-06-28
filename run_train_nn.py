from arekit.common.experiment.api.ctx_training import ExperimentTrainingContext
from arekit.common.experiment.engine import ExperimentEngine
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.contrib.networks.core.callback.hidden import HiddenStatesWriterCallback
from arekit.contrib.networks.core.callback.hidden_input import InputHiddenStatesWriterCallback
from arekit.contrib.networks.core.callback.stat import TrainingStatProviderCallback
from arekit.contrib.networks.core.callback.train_limiter import TrainingLimiterCallback
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.contrib.networks.factory import create_network_and_network_config_funcs
from arekit.contrib.networks.handlers.training import NetworksTrainingIterationHandler
from arekit.contrib.utils.model_io.tf_networks import DefaultNetworkIOUtils
from arekit.contrib.utils.np_utils.writer import NpzDataWriter

from experiment.io import CustomExperimentTrainIO
from folding.fixed import create_train_test_folding
from utils import read_train_test

if __name__ == '__main__':

    model_target_dir = "_model"
    model_load_dir = "_model/cnn"
    exp_name = "serialize"
    extra_name_suffix = "nn"
    epochs_count = 100
    labels_count = 2
    model_name = ModelNames.CNN

    train_filenames, test_filenames = read_train_test("data/split_fixed.txt")
    filenames_by_ids, data_folding = create_train_test_folding(
        train_filenames=train_filenames,
        test_filenames=test_filenames)

    model_io = NeuralNetworkModelIO(
        full_model_name="-".join([data_folding.Name, model_name.value]),
        target_dir="_out",
        model_name_tag=u'')

    exp_ctx = ExperimentTrainingContext(
        labels_count=labels_count,
        name_provider=ExperimentNameProvider(name=exp_name, suffix=extra_name_suffix),
        data_folding=data_folding)

    exp_ctx.set_model_io(model_io)

    exp_io = CustomExperimentTrainIO(exp_ctx=exp_ctx, source_dir="_out")

    data_writer = NpzDataWriter()

    network_func, network_config_func = create_network_and_network_config_funcs(
        model_name=model_name,
        model_input_type=ModelInputType.SingleInstance)

    nework_callbacks = [
        TrainingLimiterCallback(train_acc_limit=0.99),
        TrainingStatProviderCallback(),
        HiddenStatesWriterCallback(log_dir=model_target_dir, writer=data_writer),
        InputHiddenStatesWriterCallback(log_dir=model_target_dir, writer=data_writer)
    ]

    training_handler = NetworksTrainingIterationHandler(
        load_model=model_load_dir is not None,
        exp_ctx=exp_ctx,
        exp_io=exp_io,
        create_network_func=network_func,
        config=network_config_func(),
        bags_collection_type=SingleBagsCollection,
        network_callbacks=nework_callbacks,
        training_epochs=epochs_count)

    engine = ExperimentEngine()

    engine.run(states_iter=[0], handlers=[training_handler])
