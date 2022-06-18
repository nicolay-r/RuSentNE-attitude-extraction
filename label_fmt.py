from arekit.common.labels.str_fmt import StringLabelsFormatter
import labels


class CustomLabelFormatter(StringLabelsFormatter):

    def __init__(self):

        stol = {
            "OPINION_BELONGS_TO": labels.OpinionBelongsTo,
            "OPINION_RELATES_TO": labels.OpinionRelatesTo,
            "NEG_EFFECT_FROM": labels.NegEffectFrom,
            "POS_EFFECT_FROM": labels.PosEffectFrom,
            "NEG_STATE_FROM": labels.NegStateFrom,
            "POS_STATE_FROM": labels.PosStateFrom,
            "NEGATIVE_TO": labels.NegativeTo,
            "POSITIVE_TO": labels.PositiveTo,
            "STATE_BELONGS_TO": labels.StateBelongsTo,
            "POS_AUTHOR_FROM": labels.PosAuthorFrom,
            "NEG_AUTHOR_FROM": labels.NegAuthorFrom,
            "ALTERNATIVE_NAME": labels.AlternativeName,
            "ORIGINS_FROM": labels.OriginsFrom
        }

        super(CustomLabelFormatter, self).__init__(stol=stol)
