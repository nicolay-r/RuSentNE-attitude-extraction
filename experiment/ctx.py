from arekit.contrib.experiment_rusentrel.connotations.provider import RuSentiFramesConnotationProvider
from arekit.contrib.networks.core.input.ctx_serialization import NetworkSerializationContext
from arekit.contrib.networks.embedding import Embedding
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.processing.pos.base import POSTagger

from labels.scaler import ThreeLabelScaler


class CustomNetworkSerializationContext(NetworkSerializationContext):

    def __init__(self, labels_scaler, pos_tagger, embedding, terms_per_context, annotator,
                 frames_collection, frame_variant_collection, name_provider, data_folding):
        assert(isinstance(embedding, Embedding))
        assert(isinstance(pos_tagger, POSTagger))
        assert(isinstance(frames_collection, RuSentiFramesCollection))
        assert(isinstance(terms_per_context, int))

        super(CustomNetworkSerializationContext, self).__init__(
            labels_scaler=labels_scaler, annot=annotator,
            name_provider=name_provider, data_folding=data_folding)

        self.__pos_tagger = pos_tagger
        self.__terms_per_context = terms_per_context
        self.__word_embedding = embedding
        self.__frames_collection = frames_collection
        self.__frame_variant_collection = frame_variant_collection
        self.__frame_roles_label_scaler = ThreeLabelScaler()
        self.__frames_connotation_provider = RuSentiFramesConnotationProvider(collection=self.__frames_collection)

    @property
    def PosTagger(self):
        return self.__pos_tagger

    @property
    def FrameVariantCollection(self):
        return self.__frame_variant_collection

    @property
    def WordEmbedding(self):
        return self.__word_embedding

    @property
    def TermsPerContext(self):
        return self.__terms_per_context

    @property
    def FramesConnotationProvider(self):
        return self.__frames_connotation_provider

    @property
    def FrameRolesLabelScaler(self):
        return self.__frame_roles_label_scaler
