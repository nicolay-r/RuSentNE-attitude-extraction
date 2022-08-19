from arekit.common.experiment.data_type import DataType

from annot import create_neutral_annotator
from pipelines.etalon import create_etalon_pipeline, create_etalon_with_no_label_pipeline
from pipelines.test import create_test_pipeline
from pipelines.train import create_train_pipeline
from collection.annot import PredefinedTextOpinionAnnotator


def prepare_data_pipelines(text_parser, doc_ops, terms_per_context):
    """ Создаем словарь из pipelines для каждого типа данных.
    """

    train_neut_annot = create_neutral_annotator(terms_per_context)
    test_neut_annot = create_neutral_annotator(terms_per_context)

    return {
        DataType.Train: create_train_pipeline(text_parser=text_parser,
                                              doc_ops=doc_ops,
                                              annotators=[
                                                  PredefinedTextOpinionAnnotator(doc_ops),
                                                  train_neut_annot
                                              ],
                                              terms_per_context=terms_per_context),
        DataType.Test: create_test_pipeline(text_parser=text_parser,
                                            doc_ops=doc_ops,
                                            annotators=[
                                                test_neut_annot
                                            ],
                                            terms_per_context=terms_per_context),
        DataType.Etalon: create_etalon_pipeline(text_parser=text_parser,
                                                doc_ops=doc_ops,
                                                terms_per_context=terms_per_context),
        DataType.Dev: create_etalon_with_no_label_pipeline(text_parser=text_parser,
                                                           doc_ops=doc_ops,
                                                           annotators=[
                                                               PredefinedTextOpinionAnnotator(doc_ops),
                                                               train_neut_annot
                                                           ],
                                                           terms_per_context=terms_per_context)
    }
