from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.engine import ExperimentEngine
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.networks.core.input.term_types import TermTypes
from arekit.contrib.networks.handlers.serializer import NetworksInputSerializerExperimentIteration
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
from experiment.io import CustomExperimentSerializationIO
from folding.factory import FoldingFactory
from labels.formatter import SentimentLabelFormatter
from labels.scaler import PosNegNeuRelationsLabelScaler
from pipelines.collection import prepare_data_pipelines


def serialize_nn(output_dir, split_filepath, folding_type="fixed",
                 entities_fmt=CustomEntitiesFormatter(), limit=None, suffix="nn"):
    """ Run data preparation process for neural networks, i.e.
        convolutional neural networks and recurrent-based neural networks.
        Implementation based on AREkit toolkit API.
    """
    assert(isinstance(suffix, str))
    assert(isinstance(output_dir, str))
    assert(isinstance(limit, int) or limit is None)

    doc_ops = None
    data_folding = None

    if folding_type == "fixed":
        data_folding, doc_ops = FoldingFactory.create_fixed_folding(
            fixed_split_filepath=split_filepath,
            label_formatter=SentimentLabelFormatter(),
            limit=limit)

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
        labels_scaler=PosNegNeuRelationsLabelScaler(),
        embedding=load_embedding_news_mystem_skipgram_1000_20_2015(),
        terms_per_context=terms_per_context,
        pos_tagger=pos_tagger,
        name_provider=name_provider,
        frames_collection=frames_collection,
        frame_variant_collection=frame_variant_collection)

    text_parser = BaseTextParser([
        BratTextEntitiesParser(),
        DefaultTextTokenizer(keep_tokens=True),
        LemmasBasedFrameVariantsParser(frame_variants=exp_ctx.FrameVariantCollection,
                                       stemmer=stemmer)]
    )

    bpe_vectorizer = BPEVectorizer(embedding=exp_ctx.WordEmbedding, max_part_size=3)
    norm_vectorizer = RandomNormalVectorizer(vector_size=exp_ctx.WordEmbedding.VectorSize,
                                             token_offset=12345)

    handler = NetworksInputSerializerExperimentIteration(
        vectorizers={
            TermTypes.WORD: bpe_vectorizer,
            TermTypes.ENTITY: bpe_vectorizer,
            TermTypes.FRAME: bpe_vectorizer,
            TermTypes.TOKEN: norm_vectorizer
        },
        data_folding=data_folding,
        exp_io=CustomExperimentSerializationIO(output_dir=output_dir, exp_ctx=exp_ctx),
        data_type_pipelines=prepare_data_pipelines(
            text_parser=text_parser, doc_ops=doc_ops, terms_per_context=terms_per_context),
        str_entity_fmt=entities_fmt,
        balance_func=lambda data_type: data_type == DataType.Train,
        save_labels_func=lambda data_type: data_type == DataType.Train or data_type == DataType.Etalon,
        exp_ctx=exp_ctx,
        doc_ops=doc_ops,
        save_embedding=True)

    engine = ExperimentEngine()
    engine.run(states_iter=[0], handlers=[handler])
