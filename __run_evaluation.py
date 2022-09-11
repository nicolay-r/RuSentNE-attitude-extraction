import os
import unittest
from collections import OrderedDict
from os.path import join, basename, dirname

from arekit.common.experiment.data_type import DataType
from arekit.contrib.utils.evaluation.analyze_errors import extract_errors

from evaluation.calc_labels import calculate_predicted_count_per_label
from evaluation.eval_document_level import opinions_per_document_two_class_result_evaluation
from evaluation.eval_instance_level import text_opinion_per_collection_result_evaluator
from evaluation.eval_instance_level_per_doc import text_opinion_per_document_result_evaluator
from utils import show_stat_for_samples


class TestEvaluation(unittest.TestCase):

    __output_dir = "_out"

    datatypes_mapping = {
        DataType.Train: "train",
        DataType.Test: "test",
        DataType.Dev: "dev"
    }

    models = [
        # ----- nn-based
        "predict-opennre-pcnn-{}.tsv.gz",
        "predict-opennre-cnn-{}.tsv.gz",
        "predict-cnn-{}.tsv.gz",
        "predict-pcnn-{}.tsv.gz",
        "predict-att-cnn-{}.tsv.gz",
        "predict-att-pcnn-{}.tsv.gz",
        "predict-rnn-{}.tsv.gz",
        "predict-rcnn-{}.tsv.gz",
        "predict-ian-ends-{}.tsv.gz",
        "predict-self-att-bilstm-{}.tsv.gz",
        # ------ bert-based
        "predict-bert-ra-{}.tsv.gz",
        "predict-bert-ra-rsr-{}.tsv.gz",
        "predict-bert-ra-rsr-ft-{}.tsv.gz",
        "predict-bert-base-multilingual-cased-{}.tsv.gz",
        "predict-DeepPavlov-rubert-base-cased-{}.tsv.gz",
        "predict-bert-base-multilingual-cased-entity-{}.tsv.gz",
        "predict-col-DeepPavlov-rubert-base-cased-cls-{}.tsv.gz",
        "predict-ra-DeepPavlov-rubert-base-cased-cls-{}.tsv.gz",
        "predict-ra-col-DeepPavlov-rubert-base-cased-cls-{}.tsv.gz",
        "predict-ra-rsr-col-DeepPavlov-rubert-base-cased-cls-{}.tsv.gz",
    ]

    samples = {
        # for 2 classes only [test].
        "test": ["serialize-{}", "sample-test-0.tsv.gz", "sample-etalon-0.tsv.gz"],
        # for 2 and 3 classes [train].
        "train": ["serialize-{}", "sample-train-0.tsv.gz", "sample-train-0.tsv.gz"],
        # for 3 classes only [test].
        "dev": ["serialize-{}", "sample-test-0.tsv.gz", "sample-dev-0.tsv.gz"]
    }

    @staticmethod
    def show_result(total_result, evaluator_type, line_end=''):
        assert(isinstance(total_result, OrderedDict))

        data = ",".join([str(round(total_result["f1"], 4)),
                         str(round(total_result["f1_pos"], 4)),
                         str(round(total_result["f1_neg"], 4)),
                         " " if evaluator_type != "three_class" else str(round(total_result["f1_neu"], 4))
                         ])

        print(data + (',' if line_end == '' else ''), end=line_end)

    @staticmethod
    def show_acc(total_result, line_end=''):
        assert(isinstance(total_result, OrderedDict))
        data = ",".join([str(round(total_result["acc"], 4))])
        print(data + ",", end=line_end)

    @staticmethod
    def do_analysis(td_result, test_samples_filepath, etalon_samples_filepath, test_predict_filepath):
        eval_errors_df = extract_errors(eval_result=td_result,
                                        test_samples_filepath=test_samples_filepath,
                                        etalon_samples_filepath=etalon_samples_filepath)
        analysis_filename = "".join(["error-", basename(test_predict_filepath).split('.')[0], '.tsv.gz'])
        target = os.path.join(dirname(test_predict_filepath), analysis_filename)
        eval_errors_df.to_csv(target, compression='infer', sep="\t", encoding='utf-8')

    @staticmethod
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

        TestEvaluation.do_analysis(td_result=td_result,
                                   test_samples_filepath=test_samples_filepath,
                                   etalon_samples_filepath=etalon_samples_filepath,
                                   test_predict_filepath=test_predict_filepath)

        o_result = opinions_per_document_two_class_result_evaluation(
            evaluator_type=evaluator_type,
            test_predict_filepath=test_predict_filepath,
            etalon_samples_filepath=etalon_samples_filepath,
            doc_ids_mode=doc_ids_mode,
            test_samples_filepath=test_samples_filepath)

        TestEvaluation.show_result(to_result.TotalResult, evaluator_type)
        TestEvaluation.show_result(td_result.TotalResult, evaluator_type)
        TestEvaluation.show_result(o_result.TotalResult, evaluator_type, line_end="\n")

        # show_acc(to_result.TotalResult)
        # show_acc(td_result.TotalResult)
        # show_acc(o_result.TotalResult, line_end="\n")

    def __run_test(self, data_type, serialize_dir, samples_test, samples_etalon, evaluator_types, doc_ids_modes):

        for model_template in self.models:
            source_filename = model_template.format(self.datatypes_mapping[data_type])
            test_predict_filepath = join(self.__output_dir, serialize_dir, source_filename)
            etalon_samples_filepath = join(self.__output_dir, serialize_dir, samples_etalon)
            test_samples_filepath = join(self.__output_dir, serialize_dir, samples_test)

            for evaluator_type in evaluator_types:
                for doc_ids_mode in doc_ids_modes:
                    TestEvaluation.do_eval(test_samples_filepath=test_samples_filepath,
                                           test_predict_filepath=test_predict_filepath,
                                           etalon_samples_filepath=etalon_samples_filepath,
                                           evaluator_type=evaluator_type,
                                           doc_ids_mode=doc_ids_mode)

    def __test_core(self, doc_ids_modes, evaluator_types, data_types):
        for data_type in data_types:
            serialize_dir, samples_test, samples_etalon = self.samples[self.datatypes_mapping[data_type]]
            serialize_dir = serialize_dir.format("bert" if "bert-" in serialize_dir else "nn")

            self.__run_test(data_type=data_type,
                            doc_ids_modes=doc_ids_modes,
                            evaluator_types=evaluator_types,
                            serialize_dir=serialize_dir,
                            samples_test=samples_test,
                            samples_etalon=samples_etalon)

    def test_two_class(self):
        """ Оценка по двум классам для всех моеделей
        """
        doc_ids_modes = ["etalon"]
        evaluator_types = ["two_class"]
        data_types = [DataType.Train, DataType.Test]
        self.__test_core(doc_ids_modes=doc_ids_modes, evaluator_types=evaluator_types, data_types=data_types)

    def test_three_class(self):
        doc_ids_modes = ["etalon"]
        evaluator_types = ["three_class"]
        data_types = [DataType.Train, DataType.Dev]
        self.__test_core(doc_ids_modes=doc_ids_modes, evaluator_types=evaluator_types, data_types=data_types)

    def test_collections_stat(self):

        show_stat_for_samples(samples_filepath=join(self.__output_dir, "serialize-nn", "sample-train-0.tsv.gz"), no_label_uint=0)
        show_stat_for_samples(samples_filepath=join(self.__output_dir, "serialize-nn", "sample-test-0.tsv.gz"), no_label_uint=0)
        show_stat_for_samples(samples_filepath=join(self.__output_dir, "serialize-nn", "sample-etalon-0.tsv.gz"), no_label_uint=0)

        # show_stat_for_samples(samples_filepath=join(self.__output_dir, "serialize-rusentrel-bert", "sample-train-0.tsv.gz"), no_label_uint=0)
        # show_stat_for_samples(samples_filepath=join(self.__output_dir, "serialize-rusentrel-bert", "sample-test-0.tsv.gz"), no_label_uint=0)
        # show_stat_for_samples(samples_filepath=join(self.__output_dir, "serialize-rusentrel-bert", "sample-etalon-0.tsv.gz"), no_label_uint=0)

        # show_stat_for_samples(samples_filepath=join(self.__output_dir, "serialize-bert", "sample-train-0.tsv.gz"), no_label_uint=0)
        # show_stat_for_samples(samples_filepath=join(self.__output_dir, "serialize-bert", "sample-test-0.tsv.gz"), no_label_uint=0)
        # show_stat_for_samples(samples_filepath=join(self.__output_dir, "serialize-bert", "sample-etalon-0.tsv.gz"), no_label_uint=0)


if __name__ == '__main__':
    unittest.main()
