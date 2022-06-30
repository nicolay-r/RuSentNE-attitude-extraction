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

from annot import create_neutral_annotator
from entity.formatter import CustomEntitiesFormatter

from experiment.ctx import CustomNetworkSerializationContext
from experiment.doc_ops import CustomDocOperations
from experiment.io import CustomExperimentSerializationIO

from folding.fixed import create_train_test_folding
from labels.formatter import SentimentLabelFormatter
from labels.scaler import PosNegNeuRelationsLabelScaler
from pipelines.test import create_test_pipeline
from pipelines.train import create_train_pipeline
from utils import read_train_test


def serialize_nn(suffix, limit=None):
    """ Run data preparation process for neural networks, i.e.
        convolutional neural networks and recurrent-based neural networks.
    """
    assert(isinstance(suffix, str))
    assert(isinstance(limit, int) or limit is None)

    train_filenames, test_filenames = read_train_test("data/split_fixed.txt")
    if limit is not None:
        train_filenames = train_filenames[:limit]
        test_filenames = test_filenames[:limit]

    filenames_by_ids, data_folding = create_train_test_folding(train_filenames=train_filenames,
                                                               test_filenames=test_filenames)

    print("Documents count:", len(filenames_by_ids))

    terms_per_context = 50
    stemmer = MystemWrapper()
    label_formatter = SentimentLabelFormatter()
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

    exp_ctx = CustomNetworkSerializationContext(
       labels_scaler=PosNegNeuRelationsLabelScaler(),
       embedding=load_embedding_news_mystem_skipgram_1000_20_2015(),
       annotator=None,
       terms_per_context=terms_per_context,
       str_entity_formatter=CustomEntitiesFormatter(),
       pos_tagger=pos_tagger,
       name_provider=ExperimentNameProvider(name="serialize", suffix=suffix),
       frames_collection=frames_collection,
       frame_variant_collection=frame_variant_collection,
       data_folding=data_folding)

    text_parser = BaseTextParser([
        BratTextEntitiesParser(),
        DefaultTextTokenizer(keep_tokens=True),
        LemmasBasedFrameVariantsParser(frame_variants=exp_ctx.FrameVariantCollection,
                                       stemmer=stemmer)]
    )

    doc_ops = CustomDocOperations(exp_ctx=exp_ctx,
                                  label_formatter=label_formatter,
                                  filename_by_id=filenames_by_ids)

    bpe_vectorizer = BPEVectorizer(embedding=exp_ctx.WordEmbedding, max_part_size=3)
    norm_vectorizer = RandomNormalVectorizer(vector_size=exp_ctx.WordEmbedding.VectorSize,
                                             token_offset=12345)

    train_neut_annot, train_synonyms = create_neutral_annotator(terms_per_context)
    test_neut_annot, test_synonyms = create_neutral_annotator(terms_per_context)

    handler = NetworksInputSerializerExperimentIteration(
        balance=True,
        vectorizers={
            TermTypes.WORD: bpe_vectorizer,
            TermTypes.ENTITY: bpe_vectorizer,
            TermTypes.FRAME: bpe_vectorizer,
            TermTypes.TOKEN: norm_vectorizer
        },
        exp_io=CustomExperimentSerializationIO(output_dir="_out", exp_ctx=exp_ctx),
        data_type_pipelines={
           DataType.Train: create_train_pipeline(text_parser=text_parser,
                                                 doc_ops=doc_ops,
                                                 neut_annotator=train_neut_annot,
                                                 synonyms=train_synonyms,
                                                 terms_per_context=terms_per_context),
           DataType.Test: create_test_pipeline(text_parser=text_parser,
                                               doc_ops=doc_ops,
                                               neut_annotator=test_neut_annot,
                                               synonyms=test_synonyms,
                                               terms_per_context=terms_per_context)
        },
        save_labels_func=lambda data_type: data_type == DataType.Train,
        exp_ctx=exp_ctx,
        doc_ops=doc_ops)

    engine = ExperimentEngine()
    engine.run(states_iter=[0], handlers=[handler])


if __name__ == '__main__':
    serialize_nn(suffix="nn")
