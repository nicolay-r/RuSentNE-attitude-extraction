from collections import OrderedDict

from arekit.common.labels.base import Label, NoLabel
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.labels.scaler.sentiment import SentimentLabelScaler
from arekit.contrib.source.common.labels import NegativeLabel, PositiveLabel
import labels


class CustomLabelScaler(BaseLabelScaler):

    def __init__(self):

        self.__uint_to_label_dict = OrderedDict([
            (labels.OpinionBelongsTo(), 0),
            (labels.OpinionRelatesTo(), 1),
            (labels.NegEffectFrom(), 2),
            (labels.PosEffectFrom(), 3),
            (labels.NegStateFrom(), 4),
            (labels.PosStateFrom(), 5),
            (labels.NegativeTo(), 6),
            (labels.PositiveTo(), 7),
            (labels.StateBelongsTo(), 8)
        ])

        super(CustomLabelScaler, self).__init__(int_dict=self.__uint_to_label_dict,
                                                uint_dict=self.__uint_to_label_dict)


class ThreeLabelScaler(SentimentLabelScaler):
    """ For frames annotation
    """

    def __init__(self):

        uint_labels = [(NoLabel(), 0),
                       (PositiveLabel(), 1),
                       (NegativeLabel(), 2)]

        int_labels = [(NoLabel(), 0),
                      (PositiveLabel(), 1),
                      (NegativeLabel(), -1)]

        super(ThreeLabelScaler, self).__init__(uint_dict=OrderedDict(uint_labels),
                                               int_dict=OrderedDict(int_labels))

    def invert_label(self, label):
        int_label = self.label_to_int(label)
        return self.int_to_label(-int_label)
