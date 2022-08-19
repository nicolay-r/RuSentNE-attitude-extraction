from arekit.contrib.utils.pipelines.text_opinion.annot.predefined import PredefinedTextOpinionAnnotator

from pipelines.train import create_train_pipeline


def create_etalon_pipeline(text_parser, doc_ops, text_opinion_filters):
    """ We adopt excact the same pipeline as for training data,
        but we do not perform "NoLabel" annotation.
        (we are interested only in sentiment attitudes).
    """
    return create_train_pipeline(text_parser=text_parser,
                                 doc_ops=doc_ops,
                                 annotators=[PredefinedTextOpinionAnnotator(doc_ops)],
                                 text_opinion_filters=text_opinion_filters)


def create_etalon_with_no_label_pipeline(annotators, text_parser, doc_ops, text_opinion_filters):
    """ We adopt excact the same pipeline as for training data.
    """
    return create_train_pipeline(text_parser=text_parser,
                                 doc_ops=doc_ops,
                                 annotators=annotators,
                                 text_opinion_filters=text_opinion_filters)
