import os
from collections import OrderedDict
from os.path import join, basename, dirname

from arekit.contrib.utils.evaluation.analyze_errors import extract_errors

from evaluation.calc_labels import calculate_predicted_count_per_label
from evaluation.eval_document_level import opinions_per_document_two_class_result_evaluation
from evaluation.eval_instance_level import text_opinion_per_collection_result_evaluator
from evaluation.eval_instance_level_per_doc import text_opinion_per_document_result_evaluator
from utils import show_stat_for_samples


def show_result(total_result, evaluator_type, line_end=''):
    assert(isinstance(total_result, OrderedDict))

    data = ",".join([str(round(total_result["f1"], 4)),
                     str(round(total_result["f1_pos"], 4)),
                     str(round(total_result["f1_neg"], 4)),
                     " " if evaluator_type != "three_class" else str(round(total_result["f1_neu"], 4))
                     ])

    print(data + (',' if line_end == '' else ''), end=line_end)


def show_acc(total_result, line_end=''):
    assert(isinstance(total_result, OrderedDict))
    data = ",".join([str(round(total_result["acc"], 4))])
    print(data + ",", end=line_end)


def do_analysis(td_result):
    eval_errors_df = extract_errors(eval_result=td_result,
                                    test_samples_filepath=test_samples_filepath,
                                    etalon_samples_filepath=etalon_samples_filepath)
    analysis_filename = "".join(["error-", basename(test_predict_filepath).split('.')[0], '.tsv.gz'])
    target = os.path.join(dirname(test_predict_filepath), analysis_filename)
    eval_errors_df.to_csv(target, compression='infer', sep="\t", encoding='utf-8')


def do_eval(evaluator_type, test_predict_filepath, etalon_samples_filepath, test_samples_filepath, doc_ids_mode):
    assert(isinstance(doc_ids_mode, str))

    if not os.path.exists(test_samples_filepath) or \
        not os.path.exists(etalon_samples_filepath) or \
            not os.path.exists(test_predict_filepath):
        return

    print("Evaluate for [{predict}], using {dataset}".format(
        predict=test_predict_filepath, dataset=etalon_samples_filepath))

    labels_stat = calculate_predicted_count_per_label(test_predict_filepath)
    print("Labels stat:")
    print(list(sorted(labels_stat.items(), key=lambda item: item[0])))

    to_result = text_opinion_per_collection_result_evaluator(
        evaluator_type=evaluator_type,
        test_predict_filepath=test_predict_filepath,
        etalon_samples_filepath=etalon_samples_filepath,
        test_samples_filepath=test_samples_filepath)

    td_result = text_opinion_per_document_result_evaluator(
        evaluator_type=evaluator_type,
        doc_ids_mode=doc_ids_mode,
        test_predict_filepath=test_predict_filepath,
        etalon_samples_filepath=etalon_samples_filepath,
        test_samples_filepath=test_samples_filepath)

    do_analysis(td_result)

    o_result = opinions_per_document_two_class_result_evaluation(
        evaluator_type=evaluator_type,
        test_predict_filepath=test_predict_filepath,
        etalon_samples_filepath=etalon_samples_filepath,
        doc_ids_mode=doc_ids_mode,
        test_samples_filepath=test_samples_filepath)

    show_result(to_result.TotalResult, evaluator_type)
    show_result(td_result.TotalResult, evaluator_type)
    show_result(o_result.TotalResult, evaluator_type, line_end="\n")

    # show_acc(to_result.TotalResult)
    # show_acc(td_result.TotalResult)
    # show_acc(o_result.TotalResult, line_end="\n")


if __name__ == '__main__':

    output_dir = "_out"

    doc_ids_modes = [
        # "joined",
        "etalon",
    ]

    evaluator_types = [
        "two_class",
        # "three_class"
    ]

    data_test = [
        # ("predict-opennre-pcnn-test.tsv.gz",            "serialize-nn", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        # ("predict-opennre-cnn-test.tsv.gz",             "serialize-nn", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        # ("predict-cnn-test.tsv.gz",                     "serialize-nn", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        # ("predict-pcnn-test.tsv.gz",                    "serialize-nn", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        # ("predict-att-cnn-test.tsv.gz",                 "serialize-nn", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        # ("predict-att-pcnn-test.tsv.gz",                "serialize-nn", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        # ("predict-rnn-test.tsv.gz",                     "serialize-nn", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        # ("predict-rcnn-test.tsv.gz",                    "serialize-nn", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        # ("predict-ian-ends-test.tsv.gz",                "serialize-nn", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        # ("predict-self-att-bilstm-test.tsv.gz",         "serialize-nn", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        # ------
        # ("predict-bert-ra-test.tsv.gz",                 "serialize-bert", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        # ("predict-bert-ra-rsr-test.tsv.gz",             "serialize-bert", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        # ("predict-bert-ra-rsr-ft-test.tsv.gz",          "serialize-bert", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        # ("predict-bert-base-multilingual-cased-test.tsv.gz",   "serialize-bert", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        # ("predict-DeepPavlov-rubert-base-cased-test.tsv.gz",   "serialize-bert", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        # ("predict-bert-base-multilingual-cased-entity-test.tsv.gz", "serialize-bert", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        ("predict-col-DeepPavlov-rubert-base-cased-cls-test.tsv.gz", "serialize-bert", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        # ("predict-ra-DeepPavlov-rubert-base-cased-cls-test.tsv.gz", "serialize-bert", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        # ("predict-ra-col-DeepPavlov-rubert-base-cased-cls-test.tsv.gz", "serialize-bert", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"),
        # ("predict-ra-rsr-col-DeepPavlov-rubert-base-cased-cls-test.tsv.gz", "serialize-bert", "sample-test-0.tsv.gz",
        #  "sample-etalon-0.tsv.gz"),
    ]

    data_train = [
        # ("predict-opennre-pcnn-train.tsv.gz",     "serialize-nn", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        # ("predict-opennre-cnn-train.tsv.gz",      "serialize-nn", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        # ("predict-cnn-train.tsv.gz",              "serialize-nn", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        # ("predict-pcnn-train.tsv.gz",             "serialize-nn", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        # ("predict-att-cnn-train.tsv.gz",          "serialize-nn", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        # ("predict-att-pcnn-train.tsv.gz",         "serialize-nn", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        # ("predict-rnn-train.tsv.gz",              "serialize-nn", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        # ("predict-rcnn-train.tsv.gz",             "serialize-nn", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        # ("predict-ian-ends-train.tsv.gz",         "serialize-nn", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        # ("predict-self-att-bilstm-train.tsv.gz",  "serialize-nn", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        # # ------
        # ("predict-bert-ra-train.tsv.gz",         "serialize-bert", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        # ("predict-bert-ra-rsr-train.tsv.gz",     "serialize-bert", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        # ("predict-bert-ra-rsr-ft-train.tsv.gz",  "serialize-bert", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        # ("predict-bert-base-multilingual-cased-train.tsv.gz", "serialize-bert", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        # ("predict-DeepPavlov-rubert-base-cased-train.tsv.gz", "serialize-bert", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        # ("predict-bert-base-multilingual-cased-entity-train.tsv.gz", "serialize-bert", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        ("predict-col-DeepPavlov-rubert-base-cased-cls-train.tsv.gz", "serialize-bert", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        # ("predict-ra-DeepPavlov-rubert-base-cased-cls-train.tsv.gz", "serialize-bert", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        # ("predict-ra-col-DeepPavlov-rubert-base-cased-cls-train.tsv.gz", "serialize-bert", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"),
        # ("predict-ra-rsr-col-DeepPavlov-rubert-base-cased-cls-train.tsv.gz", "serialize-bert", "sample-train-0.tsv.gz",
        #  "sample-train-0.tsv.gz"),
    ]

    data_dev = [
        # ("predict-opennre-pcnn-test.tsv.gz",     "serialize-nn", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-opennre-cnn-test.tsv.gz",      "serialize-nn", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-cnn-test.tsv.gz",              "serialize-nn", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-pcnn-test.tsv.gz",             "serialize-nn", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-att-cnn-test.tsv.gz",          "serialize-nn", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-att-pcnn-test.tsv.gz",         "serialize-nn", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-rnn-test.tsv.gz",              "serialize-nn", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-rcnn-test.tsv.gz",             "serialize-nn", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-ian-ends-test.tsv.gz",         "serialize-nn", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-self-att-bilstm-test.tsv.gz",  "serialize-nn", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # # # ------
        # ("predict-bert-ra-test.tsv.gz",                                 "serialize-bert", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-bert-ra-rsr-test.tsv.gz",                             "serialize-bert", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-bert-ra-ft-test.tsv.gz",                              "serialize-bert", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-bert-ra-rsr-ft-test.tsv.gz",                          "serialize-bert", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-bert-base-multilingual-cased-test.tsv.gz",            "serialize-bert", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-DeepPavlov-rubert-base-cased-test.tsv.gz",            "serialize-bert", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-bert-base-multilingual-cased-entity-test.tsv.gz",     "serialize-bert", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-DeepPavlov-rubert-base-cased-entity-test.tsv.gz",     "serialize-bert", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-ra-DeepPavlov-rubert-base-cased-cls-test.tsv.gz",     "serialize-bert", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-ra-col-DeepPavlov-rubert-base-cased-cls-test.tsv.gz", "serialize-bert", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
        # ("predict-ra-rsr-col-DeepPavlov-rubert-base-cased-cls-test.tsv.gz", "serialize-bert", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"),
    ]

    show_stat_for_samples(samples_filepath=join(output_dir, "serialize-nn", "sample-train-0.tsv.gz"), no_label_uint=0)
    show_stat_for_samples(samples_filepath=join(output_dir, "serialize-nn", "sample-test-0.tsv.gz"), no_label_uint=0)
    show_stat_for_samples(samples_filepath=join(output_dir, "serialize-nn", "sample-etalon-0.tsv.gz"), no_label_uint=0)

    show_stat_for_samples(samples_filepath=join(output_dir, "serialize-rusentrel-bert", "sample-train-0.tsv.gz"), no_label_uint=0)
    show_stat_for_samples(samples_filepath=join(output_dir, "serialize-rusentrel-bert", "sample-test-0.tsv.gz"), no_label_uint=0)
    show_stat_for_samples(samples_filepath=join(output_dir, "serialize-rusentrel-bert", "sample-etalon-0.tsv.gz"), no_label_uint=0)

    show_stat_for_samples(samples_filepath=join(output_dir, "serialize-bert", "sample-train-0.tsv.gz"), no_label_uint=0)
    show_stat_for_samples(samples_filepath=join(output_dir, "serialize-bert", "sample-test-0.tsv.gz"), no_label_uint=0)
    show_stat_for_samples(samples_filepath=join(output_dir, "serialize-bert", "sample-etalon-0.tsv.gz"), no_label_uint=0)

    for source_filename, serialize_dir, samples_test, samples_etalon in data_test + data_train + data_dev:
        test_predict_filepath = join(output_dir, serialize_dir, source_filename)
        etalon_samples_filepath = join(output_dir, serialize_dir, samples_etalon)
        test_samples_filepath = join(output_dir, serialize_dir, samples_test)

        for evaluator_type in evaluator_types:
            for doc_ids_mode in doc_ids_modes:
                do_eval(test_samples_filepath=test_samples_filepath,
                        test_predict_filepath=test_predict_filepath,
                        etalon_samples_filepath=etalon_samples_filepath,
                        evaluator_type=evaluator_type,
                        doc_ids_mode=doc_ids_mode)
