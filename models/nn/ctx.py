from arekit.contrib.networks.core.input.ctx_serialization import NetworkSerializationContext
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.utils.connotations.rusentiframes_sentiment import RuSentiFramesConnotationProvider
from arekit.contrib.utils.processing.pos.base import POSTagger

from SentiNEREL.labels.scaler import ThreeLabelScaler


class CustomNetworkSerializationContext(NetworkSerializationContext):

    def __init__(self, labels_scaler, pos_tagger, frames_collection, frame_variant_collection):
        assert(isinstance(pos_tagger, POSTagger))
        assert(isinstance(frames_collection, RuSentiFramesCollection))

        super(CustomNetworkSerializationContext, self).__init__(labels_scaler=labels_scaler)

        self.__pos_tagger = pos_tagger
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
    def FramesConnotationProvider(self):
        return self.__frames_connotation_provider

    @property
    def FrameRolesLabelScaler(self):
        return self.__frame_roles_label_scaler
