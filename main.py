from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.engine import ExperimentEngine
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.fixed import FixedFolding
from arekit.common.frames.variants.collection import FrameVariantsCollection
from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.opinions.annot.algo.pair_based import PairBasedAnnotationAlgorithm
from arekit.common.opinions.annot.default import DefaultAnnotator
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.text.parser import BaseTextParser

from arekit.contrib.networks.handlers.serializer import NetworksInputSerializerExperimentIteration
from arekit.contrib.source.brat.entities.parser import BratTextEntitiesParser
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.utils.entities.formatters.str_rus_nocased_fmt import RussianEntitiesFormatter
from arekit.contrib.utils.pipelines.annot.base import attitude_extraction_default_pipeline
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection

from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.pos.mystem_wrap import POSMystemWrapper
from arekit.processing.text.pipeline_frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.processing.text.pipeline_tokenizer import DefaultTextTokenizer

from collection.entities import CollectionEntityCollection
from collection.io_utils import CollectionIOUtils
from doc_ops import CustomDocOperations
from embedding import RusvectoresEmbedding
from exp_ctx import CustomNetworkSerializationContext
from exp_io import CustomExperimentSerializationIO
from label_fmt import CustomLabelFormatter
from label_scaler import CustomLabelScaler
from pipeline import text_opinions_to_opinion_linkages_pipeline


if __name__ == '__main__':

     doc_ids = list(CollectionIOUtils.iter_collection_indices())

     print("Documents count:", len(doc_ids))

     terms_per_context = 50
     stemmer = MystemWrapper()
     label_formatter = CustomLabelFormatter()
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

     middle = int(len(doc_ids) / 2)

     data_folding = FixedFolding.from_parts(
         {DataType.Train: doc_ids[:middle], DataType.Test: doc_ids[middle:]})

     embedding = RusvectoresEmbedding.from_word2vec_format(
         filepath="data/news_mystem_skipgram_1000_20_2015.bin.gz", binary=True)
     embedding.set_stemmer(stemmer)

     exp_ctx = CustomNetworkSerializationContext(
        labels_scaler=CustomLabelScaler(),
        embedding=embedding,
        annotator=None,
        terms_per_context=terms_per_context,
        str_entity_formatter=RussianEntitiesFormatter(),
        pos_tagger=pos_tagger,
        name_provider=ExperimentNameProvider(name="serialize", suffix="nn"),
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
                                   doc_ids=doc_ids)

     test_synonyms = StemmerBasedSynonymCollection(iter_group_values_lists=[],
                                                   stemmer=MystemWrapper(),
                                                   is_read_only=False,
                                                   debug=False)

     # This is a pipeline for training data annotation.
     train_pipeline = text_opinions_to_opinion_linkages_pipeline(
         terms_per_context=terms_per_context,
         get_doc_func=lambda doc_id: doc_ops.get_doc(doc_id),
         text_parser=text_parser)

     # This is a pipeline for TEST data annotation.
     # We perform annotation of the attitudes.
     test_pipeline = attitude_extraction_default_pipeline(
         annotator=DefaultAnnotator(
             annot_algo=PairBasedAnnotationAlgorithm(
                 dist_in_terms_bound=50,
                 label_provider=ConstantLabelProvider(NoLabel())),
             create_empty_collection_func=lambda: OpinionCollection(
                 opinions=[],
                 synonyms=test_synonyms,
                 error_on_duplicates=True,
                 error_on_synonym_end_missed=False),
             get_doc_etalon_opins_func=lambda _: []),
         data_type=DataType.Test,
         get_doc_func=lambda doc_id: doc_ops.get_doc(doc_id),
         text_parser=text_parser,
         value_to_group_id_func=lambda value: CollectionEntityCollection.get_synonym_group_index_or_add(
             synonyms=test_synonyms, value=value),
         terms_per_context=50,
         entity_index_func=lambda brat_entity: brat_entity.ID)

     handler = NetworksInputSerializerExperimentIteration(
         balance=True,
         exp_io=CustomExperimentSerializationIO(output_dir="out", exp_ctx=exp_ctx),
         data_type_pipelines={
            DataType.Train: train_pipeline,
            DataType.Test: test_pipeline
         },
         save_labels_func=lambda data_type: data_type == DataType.Train,
         exp_ctx=exp_ctx,
         doc_ops=doc_ops)

     engine = ExperimentEngine()
     engine.run(states_iter=[0], handlers=[handler])
