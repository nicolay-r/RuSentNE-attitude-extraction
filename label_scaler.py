from collections import OrderedDict

from arekit.common.labels.scaler.base import BaseLabelScaler

from collection import labels


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
