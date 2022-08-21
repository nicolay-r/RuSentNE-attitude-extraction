from arekit.common.experiment.data_type import DataType
from arekit.contrib.utils.pipelines.text_opinion.annot.predefined import PredefinedTextOpinionAnnotator
from arekit.contrib.utils.pipelines.text_opinion.filters.distance_based import DistanceLimitedTextOpinionFilter
from arekit.contrib.utils.pipelines.text_opinion.filters.entity_based import EntityBasedTextOpinionFilter

from annot import create_neutral_annotator
from entity.filter import CollectionEntityFilter
from pipelines.etalon import create_etalon_pipeline, create_etalon_with_no_label_pipeline
from pipelines.test import create_test_pipeline
from pipelines.train import create_train_pipeline


def prepare_data_pipelines(text_parser, doc_ops, label_formatter, terms_per_context):
    """ Создаем словарь из pipelines для каждого типа данных.
    """

    train_neut_annot = create_neutral_annotator(terms_per_context)
    test_neut_annot = create_neutral_annotator(terms_per_context)

    text_opinion_filters = [
        EntityBasedTextOpinionFilter(entity_filter=CollectionEntityFilter()),
        DistanceLimitedTextOpinionFilter(terms_per_context)
    ]

    predefined_annot = PredefinedTextOpinionAnnotator(doc_ops, label_formatter)

    return {
        DataType.Train: create_train_pipeline(text_parser=text_parser,
                                              doc_ops=doc_ops,
                                              annotators=[
                                                  predefined_annot,
                                                  train_neut_annot
                                              ],
                                              text_opinion_filters=text_opinion_filters),
        DataType.Test: create_test_pipeline(text_parser=text_parser,
                                            doc_ops=doc_ops,
                                            annotators=[
                                                test_neut_annot
                                            ],
                                            text_opinion_filters=text_opinion_filters),
        DataType.Etalon: create_etalon_pipeline(text_parser=text_parser,
                                                doc_ops=doc_ops,
                                                predefined_annot=predefined_annot,
                                                text_opinion_filters=text_opinion_filters),
        DataType.Dev: create_etalon_with_no_label_pipeline(text_parser=text_parser,
                                                           doc_ops=doc_ops,
                                                           annotators=[
                                                               predefined_annot,
                                                               train_neut_annot
                                                           ],
                                                           text_opinion_filters=text_opinion_filters)
    }
