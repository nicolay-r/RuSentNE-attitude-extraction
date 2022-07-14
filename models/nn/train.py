from os.path import join

from arekit.common.experiment.api.ctx_training import ExperimentTrainingContext
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.networks.core.callback.hidden import HiddenStatesWriterCallback
from arekit.contrib.networks.core.callback.hidden_input import InputHiddenStatesWriterCallback
from arekit.contrib.networks.core.callback.stat import TrainingStatProviderCallback
from arekit.contrib.networks.core.callback.train_limiter import TrainingLimiterCallback
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.contrib.networks.factory import create_network_and_network_config_funcs
from arekit.contrib.networks.pipelines.items.training import NetworksTrainingPipelineItem
from arekit.contrib.utils.np_utils.writer import NpzDataWriter
from arekit.processing.languages.ru.pos_service import PartOfSpeechTypesService

from experiment.io import CustomExperimentTrainIO
from folding.factory import FoldingFactory


def train_nn(output_dir, model_log_dir, split_filepath, folding_type="fixed",
             exp_name="serialize", extra_name_suffix="nn",
             epochs_count=100, labels_count=3, model_name=ModelNames.CNN,
             bags_per_minibatch=4, bag_size=1, terms_per_context=50,
             learning_rate=0.01, embedding_dropout_keep_prob=1.0,
             dropout_keep_prob=0.9, train_acc_limit=0.99, finetune_existed=True):
    """ Training TensorFlow-based model (version 1.14),
        provided by contributional part of the AREkit framework.
        From the list of the predefined moels.

        finetune_existed: bool
            flag that shows wheter there is need to keep saved state and proceed with its finetunning.
    """
    assert(isinstance(output_dir, str))
    assert(isinstance(finetune_existed, bool))

    full_model_name = "-".join([folding_type, model_name.value])
    model_target_dir = join(model_log_dir, full_model_name)

    model_io = NeuralNetworkModelIO(full_model_name=full_model_name,
                                    target_dir=output_dir,
                                    source_dir=output_dir if finetune_existed else None,
                                    model_name_tag=u'')

    exp_ctx = ExperimentTrainingContext(
        labels_count=labels_count,
        name_provider=ExperimentNameProvider(name=exp_name, suffix=extra_name_suffix))

    exp_ctx.set_model_io(model_io)

    exp_io = CustomExperimentTrainIO(exp_ctx=exp_ctx,
                                     source_dir=output_dir)

    data_writer = NpzDataWriter()

    network_func, network_config_func = create_network_and_network_config_funcs(
        model_name=model_name,
        model_input_type=ModelInputType.SingleInstance)

    network_callbacks = [
        TrainingLimiterCallback(train_acc_limit=train_acc_limit),
        TrainingStatProviderCallback(),
        HiddenStatesWriterCallback(log_dir=model_target_dir, writer=data_writer),
        InputHiddenStatesWriterCallback(log_dir=model_target_dir, writer=data_writer)
    ]

    # Configuration initialization.
    config = network_config_func()
    config.modify_classes_count(value=labels_count)
    config.modify_learning_rate(learning_rate)
    config.modify_use_class_weights(True)
    config.modify_dropout_keep_prob(dropout_keep_prob)
    config.modify_bag_size(bag_size)
    config.modify_bags_per_minibatch(bags_per_minibatch)
    config.modify_embedding_dropout_keep_prob(embedding_dropout_keep_prob)
    config.modify_terms_per_context(terms_per_context)
    config.modify_use_entity_types_in_embedding(False)
    config.set_pos_count(PartOfSpeechTypesService.get_mystem_pos_count())

    pipeline_item = NetworksTrainingPipelineItem(
        load_model=True,
        exp_ctx=exp_ctx,
        exp_io=exp_io,
        create_network_func=network_func,
        config=config,
        bags_collection_type=SingleBagsCollection,
        network_callbacks=network_callbacks,
        training_epochs=epochs_count)

    # Start training process.
    data_folding = None
    if folding_type == "fixed":
        _, data_folding = FoldingFactory.create_fixed_folding(split_filepath)

    ppl = BasePipeline([pipeline_item])
    ppl.run(None, params_dict={"data_folding": data_folding,
                               "data_type": DataType.Train})
