from collection.annot import PredefinedTextOpinionAnnotator
from pipelines.train import create_train_pipeline


def create_etalon_pipeline(text_parser, doc_ops, terms_per_context):
    """ We adopt excact the same pipeline as for training data,
        but we do not perform "NoLabel" annotation.
        (we are interested only in sentiment attitudes).
    """
    return create_train_pipeline(text_parser=text_parser,
                                 doc_ops=doc_ops,
                                 annotators=[PredefinedTextOpinionAnnotator(doc_ops)],
                                 terms_per_context=terms_per_context)


def create_etalon_with_no_label_pipeline(annotators, text_parser, doc_ops, terms_per_context):
    """ We adopt excact the same pipeline as for training data.
    """
    return create_train_pipeline(text_parser=text_parser,
                                 doc_ops=doc_ops,
                                 annotators=annotators,
                                 terms_per_context=terms_per_context)
