from os.path import join

from evaluation.document_level import opinions_per_document_two_class_result_evaluation
from evaluation.instance_level import text_opinion_monolith_collection_two_class_result_evaluator

if __name__ == '__main__':

    output_dir = "_out"
    source_filename = "predict-cnn.tsv.gz"
    samples_test = "sample-test-0.tsv.gz"
    samples_etalon = "sample-etalon-0.tsv.gz"
    serialize_dir = "serialize-nn_3l"

    test_predict_filepath = join(output_dir, serialize_dir, source_filename)
    etalon_samples_filepath = join(output_dir, serialize_dir, samples_etalon)
    test_samples_filepath = join(output_dir, serialize_dir, samples_test)

    to_result = text_opinion_monolith_collection_two_class_result_evaluator(
        test_predict_filepath=test_predict_filepath,
        etalon_samples_filepath=etalon_samples_filepath,
        test_samples_filepath=test_samples_filepath)

    o_result = opinions_per_document_two_class_result_evaluation(
        test_predict_filepath=test_predict_filepath,
        etalon_samples_filepath=etalon_samples_filepath,
        test_samples_filepath=test_samples_filepath)

    print("Evaluate for", etalon_samples_filepath)
    print("Instance-based evaluation result:")
    print(to_result.TotalResult)
    print("Document-level-based evaluation result:")
    print(o_result.TotalResult)
