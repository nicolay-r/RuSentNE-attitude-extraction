from os.path import join

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
from arekit.contrib.utils.np_utils.writer import NpzDataWriter
from arekit.processing.languages.ru.pos_service import PartOfSpeechTypesService

from experiment.io import CustomExperimentTrainIO
from folding.fixed import create_train_test_folding
from utils import read_train_test


def train_nn(output_dir, model_log_dir, split_source, extra_name_suffix="nn",
             epochs_count=100, labels_count=3, model_name=ModelNames.CNN,
             finetune_existed=True):
    """ Training TensorFlow-based model (version 1.14),
        provided by contributional part of the AREkit framework.
        From the list of the predefined moels.

        finetune_existed: bool
            flag that shows wheter there is need to keep saved state and proceed with its finetunning.

        NOTE/TODO:
        We adopt handler, but in futher handlers might be switched to pipeline items
    """
    assert(isinstance(output_dir, str))
    assert(isinstance(finetune_existed, bool))

    exp_name = "serialize"

    train_filenames, test_filenames = read_train_test(split_source)
    filenames_by_ids, data_folding, etalon_folding = create_train_test_folding(
        train_filenames=train_filenames,
        test_filenames=test_filenames)

    full_model_name = "-".join([data_folding.Name, model_name.value])
    model_target_dir = join(model_log_dir, full_model_name)

    model_io = NeuralNetworkModelIO(full_model_name=full_model_name,
                                    target_dir=output_dir,
                                    source_dir=output_dir if finetune_existed else None,
                                    model_name_tag=u'')

    exp_ctx = ExperimentTrainingContext(
        labels_count=labels_count,
        name_provider=ExperimentNameProvider(name=exp_name, suffix=extra_name_suffix),
        data_folding=data_folding)

    exp_ctx.set_model_io(model_io)

    exp_io = CustomExperimentTrainIO(exp_ctx=exp_ctx,
                                     source_dir=output_dir)

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

    # Configuration initialization.
    config = network_config_func()
    config.modify_classes_count(value=labels_count)
    config.modify_learning_rate(0.01)
    config.modify_use_class_weights(True)
    config.modify_dropout_keep_prob(0.9)
    config.modify_bag_size(1)
    config.modify_bags_per_minibatch(4)
    config.modify_embedding_dropout_keep_prob(1.0)
    config.modify_terms_per_context(50)
    config.modify_use_entity_types_in_embedding(False)
    config.set_pos_count(PartOfSpeechTypesService.get_mystem_pos_count())

    training_handler = NetworksTrainingIterationHandler(
        load_model=True,
        exp_ctx=exp_ctx,
        exp_io=exp_io,
        create_network_func=network_func,
        config=config,
        bags_collection_type=SingleBagsCollection,
        network_callbacks=nework_callbacks,
        training_epochs=epochs_count)

    engine = ExperimentEngine()

    engine.run(states_iter=[0], handlers=[training_handler])


if __name__ == '__main__':

    train_nn(output_dir="_out",
             model_log_dir="_model",
             split_source="data/split_fixed.txt")
