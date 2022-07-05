from os.path import join

from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.experiment.api.ctx_training import ExperimentTrainingContext
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.nofold import NoFolding
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.contrib.networks.core.callback.writer import PredictResultWriterCallback
from arekit.contrib.networks.core.ctx_inference import InferenceContext
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.core.model import BaseTensorflowModel
from arekit.contrib.networks.core.model_ctx import TensorflowModelContext
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO
from arekit.contrib.networks.core.pipeline.item_fit import MinibatchFittingPipelineItem
from arekit.contrib.networks.core.pipeline.item_keep_hidden import MinibatchHiddenFetcherPipelineItem
from arekit.contrib.networks.core.pipeline.item_predict import EpochLabelsPredictorPipelineItem
from arekit.contrib.networks.core.pipeline.item_predict_labeling import EpochLabelsCollectorPipelineItem
from arekit.contrib.networks.core.predict.base_writer import BasePredictWriter
from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.contrib.networks.factory import create_network_and_network_config_funcs
from arekit.contrib.networks.shapes import NetworkInputShapes
from arekit.contrib.utils.model_io.tf_networks import DefaultNetworkIOUtils
from arekit.processing.languages.ru.pos_service import PartOfSpeechTypesService

from labels.scaler import PosNegNeuRelationsLabelScaler


class InferIOUtils(DefaultNetworkIOUtils):

    def __init__(self, output_dir, exp_ctx):
        assert(isinstance(output_dir, str))
        super(InferIOUtils, self).__init__(exp_ctx=exp_ctx)
        self.__output_dir = output_dir

    def _get_experiment_sources_dir(self):
        return self.__output_dir


class TensorflowNetworkInferencePipelineItem(BasePipelineItem):

    def __init__(self, model_name, bags_collection_type, model_input_type, predict_writer,
                 data_type, bag_size, bags_per_minibatch, nn_io, labels_scaler, callbacks):
        assert(isinstance(callbacks, list))
        assert(isinstance(bag_size, int))
        assert(isinstance(predict_writer, BasePredictWriter))
        assert(isinstance(data_type, DataType))

        # Create network an configuration.
        network_func, config_func = create_network_and_network_config_funcs(
            model_name=model_name, model_input_type=model_input_type)

        # setup network and config parameters.
        self.__network = network_func()
        self.__config = config_func()
        self.__config.modify_classes_count(labels_scaler.LabelsCount)
        self.__config.modify_bag_size(bag_size)
        self.__config.modify_bags_per_minibatch(bags_per_minibatch)
        self.__config.set_class_weights([1, 1, 1])
        self.__config.set_pos_count(PartOfSpeechTypesService.get_mystem_pos_count())
        self.__config.reinit_config_dependent_parameters()

        # intialize model context.
        self.__create_model_ctx = lambda inference_ctx: TensorflowModelContext(
            nn_io=nn_io,
            network=self.__network,
            config=self.__config,
            inference_ctx=inference_ctx,
            bags_collection_type=bags_collection_type)

        self.__callbacks = callbacks + [
            PredictResultWriterCallback(labels_scaler=labels_scaler, writer=predict_writer)
        ]

        self.__writer = predict_writer
        self.__bags_collection_type = bags_collection_type
        self.__data_type = data_type

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, InferIOUtils))
        assert(isinstance(pipeline_ctx, PipelineContext))

        # Setup predicted result writer.
        tgt = pipeline_ctx.provide_or_none("predict_fp")
        full_model_name = pipeline_ctx.provide_or_none("full_model_name")

        if tgt is None:
            exp_root = join(input_data._get_experiment_sources_dir(),
                            input_data.get_experiment_folder_name())
            tgt = join(exp_root, "predict-{}.tsv.gz".format(full_model_name))

        # Update for further pipeline items.
        pipeline_ctx.update("predict_fp", tgt)

        # Fetch other required in furter information from input_data.
        samples_filepath = input_data.create_samples_writer_target(self.__data_type)
        embedding = input_data.load_embedding()
        vocab = input_data.load_vocab()

        # Setup config parameters.
        self.__config.set_term_embedding(embedding)

        inference_ctx = InferenceContext.create_empty()
        inference_ctx.initialize(
            dtypes=[self.__data_type],
            bags_collection_type=self.__bags_collection_type,
            create_samples_view_func=lambda data_type: BaseSampleStorageView(
                storage=BaseRowsStorage.from_tsv(samples_filepath),
                row_ids_provider=MultipleIDProvider()),
            has_model_predefined_state=True,
            vocab=vocab,
            labels_count=self.__config.ClassesCount,
            input_shapes=NetworkInputShapes(iter_pairs=[
                (NetworkInputShapes.FRAMES_PER_CONTEXT, self.__config.FramesPerContext),
                (NetworkInputShapes.TERMS_PER_CONTEXT, self.__config.TermsPerContext),
                (NetworkInputShapes.SYNONYMS_PER_CONTEXT, self.__config.SynonymsPerContext),
            ]),
            bag_size=self.__config.BagSize)

        # Model preparation.
        model = BaseTensorflowModel(
            context=self.__create_model_ctx(inference_ctx),
            callbacks=self.__callbacks,
            predict_pipeline=[
                EpochLabelsPredictorPipelineItem(),
                EpochLabelsCollectorPipelineItem(),
                MinibatchHiddenFetcherPipelineItem()
            ],
            fit_pipeline=[MinibatchFittingPipelineItem()])

        self.__writer.set_target(tgt)

        model.predict(do_compile=True)


def predict_nn(extra_name_suffix,
               output_dir, data_folding_name="fixed",
               model_name=ModelNames.CNN, labels_count=3):
    """ Perform inference for dataset using a pre-trained collection
        This is a pipeline-based impelementation, taken from
        the ARElight repository, see the following code for reference:
            https://github.com/nicolay-r/ARElight/blob/v0.22.0/arelight/pipelines/inference_nn.py
    """
    assert(isinstance(output_dir, str))

    exp_name = "serialize"

    full_model_name = "-".join([data_folding_name, model_name.value])
    model_io = NeuralNetworkModelIO(full_model_name=full_model_name,
                                    target_dir=output_dir,
                                    source_dir=output_dir,
                                    model_name_tag=u'')

    ppl = BasePipeline(pipeline=[
        TensorflowNetworkInferencePipelineItem(
            data_type=DataType.Test,
            bag_size=1,
            bags_per_minibatch=4,
            model_name=model_name,
            bags_collection_type=SingleBagsCollection,
            model_input_type=ModelInputType.SingleInstance,
            predict_writer=TsvPredictWriter(),
            callbacks=[],
            labels_scaler=PosNegNeuRelationsLabelScaler(),
            nn_io=model_io)
    ])

    # Hack with the training context.
    exp_ctx = ExperimentTrainingContext(
        labels_count=labels_count,
        name_provider=ExperimentNameProvider(name=exp_name, suffix=extra_name_suffix),
        data_folding=NoFolding(doc_ids_to_fold=[], supported_data_types=[DataType.Test]))

    ppl.run(InferIOUtils(output_dir=output_dir, exp_ctx=exp_ctx),
            {
                "full_model_name": model_name.value
            })
