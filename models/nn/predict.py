from os.path import join

from arekit.common.data.row_ids.multiple import MultipleIDProvider
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.data.views.samples import BaseSampleStorageView
from arekit.common.experiment.api.base_samples_io import BaseSamplesIO
from arekit.common.experiment.api.ctx_base import ExperimentContext
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.base import BaseDataFolding
from arekit.common.folding.nofold import NoFolding
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.contrib.networks.core.callback.writer import PredictResultWriterCallback
from arekit.contrib.networks.core.ctx_inference import InferenceContext
from arekit.contrib.networks.core.embedding_io import BaseEmbeddingIO
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
from arekit.contrib.utils.io_utils.embedding import NpzEmbeddingIOUtils
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.processing.languages.ru.pos_service import PartOfSpeechTypesService

from labels.scaler import PosNegNeuRelationsLabelScaler


class TensorflowNetworkInferencePipelineItem(BasePipelineItem):

    def __init__(self, model_name, bags_collection_type, model_input_type, predict_writer,
                 data_type, bag_size, bags_per_minibatch, nn_io, labels_scaler, callbacks,
                 data_folding):
        assert(isinstance(callbacks, list))
        assert(isinstance(bag_size, int))
        assert(isinstance(predict_writer, BasePredictWriter))
        assert(isinstance(data_type, DataType))
        assert(isinstance(data_folding, BaseDataFolding))

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
        self.__data_folding = data_folding

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(pipeline_ctx, PipelineContext))
        assert("emb_io" in input_data)
        assert("samples_io" in input_data)
        assert("predict_root" in input_data)

        emb_io = input_data["emb_io"]
        samples_io = input_data["samples_io"]
        predict_root = input_data["predict_root"]
        assert(isinstance(emb_io, BaseEmbeddingIO))
        assert(isinstance(samples_io, BaseSamplesIO))

        # Setup predicted result writer.
        full_model_name = pipeline_ctx.provide_or_none("full_model_name")
        tgt = join(predict_root, "predict-{fmn}-{dtype}.tsv.gz".format(
            fmn=full_model_name, dtype=str(self.__data_type).lower().split('.')[-1]))

        # Fetch other required in furter information from input_data.
        samples_filepath = samples_io.create_target(
            data_type=self.__data_type, data_folding=self.__data_folding)
        embedding = emb_io.load_embedding(data_folding=self.__data_folding)
        vocab = emb_io.load_vocab(data_folding=self.__data_folding)

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

        model.predict(data_type=self.__data_type, do_compile=True)


def predict_nn(extra_name_suffix, output_dir, embedding_dir, samples_dir, exp_name="serialize",
               data_folding_name="fixed", bag_size=1, bags_per_minibatch=4,
               model_name=ModelNames.CNN, data_type=DataType.Test):
    """ Perform inference for dataset using a pre-trained collection
        This is a pipeline-based impelementation, taken from
        the ARElight repository, see the following code for reference:
            https://github.com/nicolay-r/ARElight/blob/v0.22.0/arelight/pipelines/inference_nn.py
    """
    assert(isinstance(output_dir, str))
    assert(isinstance(embedding_dir, str))
    assert(isinstance(samples_dir, str))

    data_folding = NoFolding(doc_ids_to_fold=[], supported_data_types=[data_type])
    full_model_name = "-".join([data_folding_name, model_name.value])
    model_io = NeuralNetworkModelIO(full_model_name=full_model_name,
                                    target_dir=output_dir,
                                    source_dir=output_dir,
                                    model_name_tag=u'')

    ppl = BasePipeline(pipeline=[
        TensorflowNetworkInferencePipelineItem(
            data_type=data_type,
            bag_size=bag_size,
            bags_per_minibatch=bags_per_minibatch,
            model_name=model_name,
            bags_collection_type=SingleBagsCollection,
            model_input_type=ModelInputType.SingleInstance,
            predict_writer=TsvPredictWriter(),
            callbacks=[],
            labels_scaler=PosNegNeuRelationsLabelScaler(),
            data_folding=data_folding,
            nn_io=model_io)
    ])

    # Hack with the training context.
    exp_ctx = ExperimentContext(
        name_provider=ExperimentNameProvider(name=exp_name, suffix=extra_name_suffix))

    input_data = {
        "samples_io": SamplesIO(target_dir=samples_dir),
        "emb_io": NpzEmbeddingIOUtils(target_dir=embedding_dir, exp_ctx=exp_ctx),
        "predict_root": output_dir
    }

    ppl.run(input_data=input_data, params_dict={ "full_model_name": model_name.value })
