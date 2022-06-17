from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.engine import ExperimentEngine
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.nofold import NoFolding
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.networks.handlers.serializer import NetworksInputSerializerExperimentIteration
from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.utils.entities.formatters.str_rus_nocased_fmt import RussianEntitiesFormatter
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.pos.mystem_wrap import POSMystemWrapper
from arekit.processing.text.pipeline_frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.processing.text.pipeline_tokenizer import DefaultTextTokenizer

from doc_ops import CustomDocOperations
from embedding import RusvectoresEmbedding
from exp_ctx import CustomNetworkSerializationContext
from exp_io import CustomExperimentSerializationIO
from label_fmt import CustomLabelFormatter
from label_scaler import CustomLabelScaler
from pipeline import text_opinions_to_opinion_linkages_pipeline


if __name__ == '__main__':

     doc_ids = [15088]

     label_formatter = CustomLabelFormatter()
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

     data_folding = NoFolding(doc_ids_to_fold=doc_ids, supported_data_types=[DataType.Train])

     embedding = RusvectoresEmbedding.from_word2vec_format(
         filepath="data/news_mystem_skipgram_1000_20_2015.bin.gz", binary=True)
     embedding.set_stemmer(stemmer)

     exp_ctx = CustomNetworkSerializationContext(
        labels_scaler=CustomLabelScaler(),
        embedding=embedding,
        annotator=None,
        terms_per_context=terms_per_context,
        # str_entity_formatter=SimpleUppercasedEntityFormatter(),
        str_entity_formatter=RussianEntitiesFormatter(),
        pos_tagger=pos_tagger,
        name_provider=ExperimentNameProvider(name="serialize", suffix="nn"),
        frames_collection=frames_collection,
        frame_variant_collection=frame_variant_collection,
        data_folding=data_folding)

     handler = NetworksInputSerializerExperimentIteration(
         balance=True,
         exp_io=CustomExperimentSerializationIO(output_dir="out", exp_ctx=exp_ctx),
         pipeline=text_opinions_to_opinion_linkages_pipeline(
             label_formatter=label_formatter,
             terms_per_context=terms_per_context,
             text_parser=BaseTextParser([
                 BratTextEntitiesParser(),
                 DefaultTextTokenizer(keep_tokens=True),
                 LemmasBasedFrameVariantsParser(frame_variants=exp_ctx.FrameVariantCollection,
                                                stemmer=stemmer)]
             )),
         exp_ctx=exp_ctx,
         doc_ops=CustomDocOperations(exp_ctx=exp_ctx))

     engine = ExperimentEngine(data_folding)
     engine.run(handlers=[handler])
