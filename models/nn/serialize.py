from arekit.common.data.input.writers.base import BaseWriter
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.pipeline.base import BasePipeline
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.networks.core.input.term_types import TermTypes
from arekit.contrib.networks.pipelines.items.serializer import NetworksInputSerializerPipelineItem
from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.utils.resources import load_embedding_news_mystem_skipgram_1000_20_2015
from arekit.contrib.utils.vectorizers.bpe import BPEVectorizer
from arekit.contrib.utils.vectorizers.random_norm import RandomNormalVectorizer
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.pos.mystem_wrap import POSMystemWrapper
from arekit.processing.text.pipeline_frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.processing.text.pipeline_tokenizer import DefaultTextTokenizer

from entity.formatter import CustomEntitiesFormatter
from experiment.ctx import CustomNetworkSerializationContext
from experiment.doc_ops import CustomDocOperations
from experiment.io import CustomNetworkSerializationIO
from folding.factory import FoldingFactory
from labels.formatter import SentimentLabelFormatter
from labels.scaler import PosNegNeuRelationsLabelScaler
from pipelines.collection import prepare_data_pipelines


def serialize_nn(output_dir, split_filepath, writer=None, folding_type="fixed",
                 labels_scaler=PosNegNeuRelationsLabelScaler(),
                 entities_fmt=CustomEntitiesFormatter(), limit=None, suffix="nn"):
    """ Run data preparation process for neural networks, i.e.
        convolutional neural networks and recurrent-based neural networks.
        Implementation based on AREkit toolkit API.
    """
    assert(isinstance(suffix, str))
    assert(isinstance(output_dir, str))
    assert(isinstance(limit, int) or limit is None)
    assert(isinstance(writer, BaseWriter) or writer is None)

    terms_per_context = 50
    stemmer = MystemWrapper()
    pos_tagger = POSMystemWrapper(mystem=stemmer.MystemInstance)

    # Frames initialization
    frames_collection = RuSentiFramesCollection.read_collection(
        version=RuSentiFramesVersions.V20,
        labels_fmt=RuSentiFramesLabelsFormatter())
    frame_variant_collection = FrameVariantsCollection()
    frame_variant_collection.fill_from_iterable(
        variants_with_id=frames_collection.iter_frame_id_and_variants(),
        overwrite_existed_variant=True,
        raise_error_on_existed_variant=False)

    name_provider = ExperimentNameProvider(name="serialize", suffix=suffix)

    exp_ctx = CustomNetworkSerializationContext(
        labels_scaler=labels_scaler,
        terms_per_context=terms_per_context,
        pos_tagger=pos_tagger,
        name_provider=name_provider,
        frames_collection=frames_collection,
        frame_variant_collection=frame_variant_collection)

    embedding = load_embedding_news_mystem_skipgram_1000_20_2015()
    bpe_vectorizer = BPEVectorizer(embedding=embedding, max_part_size=3)
    norm_vectorizer = RandomNormalVectorizer(vector_size=embedding.VectorSize,
                                             token_offset=12345)

    pipeline_item = NetworksInputSerializerPipelineItem(
        vectorizers={
            TermTypes.WORD: bpe_vectorizer,
            TermTypes.ENTITY: bpe_vectorizer,
            TermTypes.FRAME: bpe_vectorizer,
            TermTypes.TOKEN: norm_vectorizer
        },
        exp_io=CustomNetworkSerializationIO(output_dir=output_dir, exp_ctx=exp_ctx, writer=writer),
        str_entity_fmt=entities_fmt,
        balance_func=lambda data_type: data_type == DataType.Train,
        save_labels_func=lambda data_type: data_type == DataType.Train or data_type == DataType.Etalon,
        exp_ctx=exp_ctx,
        save_embedding=True)

    data_folding = None
    filenames_by_ids = None

    if folding_type == "fixed":
        filenames_by_ids, data_folding = FoldingFactory.create_fixed_folding(
            fixed_split_filepath=split_filepath, limit=limit)

    doc_ops = CustomDocOperations(label_formatter=SentimentLabelFormatter(),
                                  filename_by_id=filenames_by_ids)

    text_parser = BaseTextParser([
        BratTextEntitiesParser(),
        DefaultTextTokenizer(keep_tokens=True),
        LemmasBasedFrameVariantsParser(frame_variants=exp_ctx.FrameVariantCollection,
                                       stemmer=stemmer)]
    )

    data_type_pipelines = prepare_data_pipelines(text_parser=text_parser,
                                                 doc_ops=doc_ops,
                                                 terms_per_context=terms_per_context)

    ppl = BasePipeline([pipeline_item])
    ppl.run(input_data=None,
            params_dict={
                "data_folding": data_folding,
                "data_type_pipelines": data_type_pipelines
    })
