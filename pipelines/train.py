from arekit.contrib.utils.pipelines.text_opinion.extraction import text_opinion_extraction_pipeline


def create_train_pipeline(text_parser, doc_ops, annotators, text_opinion_filters):
    """ Train pipeline is based on the predefined annotations and
        automatic annotations of other pairs with a NoLabel.
    """
    return text_opinion_extraction_pipeline(
        get_doc_func=lambda doc_id: doc_ops.get_doc(doc_id),
        text_parser=text_parser,
        annotators=annotators,
        text_opinion_filters=text_opinion_filters)
