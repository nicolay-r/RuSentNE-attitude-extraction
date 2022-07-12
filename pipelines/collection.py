from arekit.common.experiment.data_type import DataType

from annot import create_neutral_annotator
from pipelines.etalon import create_etalon_pipeline
from pipelines.test import create_test_pipeline
from pipelines.train import create_train_pipeline


def prepare_data_pipelines(text_parser, doc_ops, terms_per_context):
    """ Создаем словарь из pipelines для каждого типа данных.
    """

    train_neut_annot, train_synonyms = create_neutral_annotator(terms_per_context)
    test_neut_annot, test_synonyms = create_neutral_annotator(terms_per_context)
    _, etalon_synonyms = create_neutral_annotator(terms_per_context)

    return {
        DataType.Train: create_train_pipeline(text_parser=text_parser,
                                              doc_ops=doc_ops,
                                              annotator=train_neut_annot,
                                              synonyms=train_synonyms,
                                              terms_per_context=terms_per_context),
        DataType.Test: create_test_pipeline(text_parser=text_parser,
                                            doc_ops=doc_ops,
                                            annotator=test_neut_annot,
                                            synonyms=test_synonyms,
                                            terms_per_context=terms_per_context),
        DataType.Etalon: create_etalon_pipeline(text_parser=text_parser,
                                                doc_ops=doc_ops,
                                                synonyms=etalon_synonyms,
                                                terms_per_context=terms_per_context)
    }
