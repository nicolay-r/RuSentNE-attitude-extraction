from arekit.contrib.utils.evaluation.evaluators.three_class import ThreeClassEvaluator
from arekit.contrib.utils.evaluation.evaluators.two_class import TwoClassEvaluator


def create_evaluator(evaluator_type, comparator, label_scaler, get_item_label_func, uint_labels):
    """ TODO: #363
        https://github.com/nicolay-r/AREkit/issues/363
        This should bere removed, since we consider a MulticlassEvaluator.
        This is now limited to 2 and 3.
    """
    assert(isinstance(evaluator_type, str))
    assert(isinstance(uint_labels, list))

    if evaluator_type == "two_class":
        return TwoClassEvaluator(
            comparator=comparator,
            label1=label_scaler.uint_to_label(uint_labels[0]),
            label2=label_scaler.uint_to_label(uint_labels[1]),
            get_item_label_func=get_item_label_func)

    if evaluator_type == "three_class":
        return ThreeClassEvaluator(
            comparator=comparator,
            label1=label_scaler.uint_to_label(uint_labels[0]),
            label2=label_scaler.uint_to_label(uint_labels[1]),
            label3=label_scaler.uint_to_label(uint_labels[2]),
            get_item_label_func=get_item_label_func)


def create_filter_labels_func(evaluator_type, get_label_func, no_label):
    """ TODO: #363
        https://github.com/nicolay-r/AREkit/issues/363
        provide just labels that should be ignored instead, once #363 will providee Multiclass Evaluator.
        This is now limited to 2 and 3.
    """
    assert(callable(get_label_func))

    if evaluator_type == "two_class":
        return lambda item: get_label_func(item) != no_label
    if evaluator_type == "three_class":
        return lambda item: True
